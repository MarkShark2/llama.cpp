// [fork] DSA sparse flash attention kernel.
//
// Shape rationale (DSV4-Flash: MQA, n_head = 64, DK = DV = 512, top_k = 512):
// one CUDA block processes ONE query token and a chunk of HCHUNK query heads.
// All heads of a token share the same gathered KV rows, so each K/V row is
// staged to shared memory once per head-chunk and consumed by every head —
// unlike the dense tile kernels, whose per-Q-tile K reuse dies when every
// query token has its own top-k index list.
//
// Two segments feed one online softmax per head:
//   segment 0 (dense):    rows 0..ne_dense of src[1]/src[2] with the f16 mask
//                         (the raw SWA window in DSV4)
//   segment 1 (gathered): rows kc[idx[j,t]] for idx[j,t] in [0, nvis[t])
//                         (the top-k compressed cells; V = DV-prefix of the row)
//
// v1 constraints (checked in ggml_cuda_fattn_sparse_supported): K/V/kc F16 with
// 16-byte-aligned row strides, Q F32, DK == DV in {512}, single KV head,
// max_bias == 0, no logit softcap. Numerics follow the CPU reference:
// natural exp, fp32 accumulators, M initialized to a large negative finite
// value so empty tiles cannot produce (-inf) - (-inf) NaNs.

#include "common.cuh"
#include "fattn-sparse.cuh"

// declared in ggml.c (fork extension, not in public headers)
extern "C" bool ggml_flash_attn_ext_is_sparse(const struct ggml_tensor * a);

#define FATTN_SPARSE_TILE    32  // KV rows staged per iteration
#define FATTN_SPARSE_HCHUNK  16  // query heads per block
#define FATTN_SPARSE_NTH    256  // threads per block (16 threads per head)
#define FATTN_SPARSE_M_INIT (-1.0e30f)

typedef struct {
    const char * q;        // f32 [DK, T, H, ns]
    const char * k;        // f16 [DK, ne_dense, 1, ns]
    const char * v;        // f16 [DV, ne_dense, 1, ns]
    const char * mask;     // f16 [ne_dense(padded), T, 1, ns] or null
    const char * sinks;    // f32 [H] or null
    const char * kc;       // f16 [DK, n_cells, 1, ns]
    const char * idx;      // i32 [n_idx, T, 1, ns]
    const char * nvis;     // i32 [T, 1, 1, ns]
    float      * dst;      // f32 [DV, H, T, ns] contiguous

    float   scale;
    int     ne_dense;      // dense KV rows (src[1]->ne[1])
    int     n_idx;         // gathered candidates per token
    int     n_head;

    // byte strides
    size_t  nbq1, nbq2, nbq3;         // Q: token, head, stream
    size_t  nbk1, nbk3;               // dense K row, stream
    size_t  nbv1, nbv3;               // dense V row, stream
    size_t  nbm1, nbm3;               // mask row-of-token, stream
    size_t  nbc1, nbc3;               // kc row, stream
    size_t  nbi1, nbi3;               // idx token, stream
    size_t  nbn0, nbn3;               // nvis token, stream
} fattn_sparse_params;

template <int DK, int DV>
static __global__ void flash_attn_sparse_f16(const fattn_sparse_params p) {
    constexpr int TILE   = FATTN_SPARSE_TILE;
    constexpr int HCHUNK = FATTN_SPARSE_HCHUNK;
    constexpr int TPH    = FATTN_SPARSE_NTH/HCHUNK; // threads per head
    constexpr int DPT    = DK/TPH;                  // dims per thread (Q/K dot slice)
    constexpr int VPT    = DV/TPH;                  // dims per thread (V accumulate slice)
    constexpr int H2PT   = DPT/2;                   // half2 per thread per row
    static_assert(DK == DV, "the strided lane mapping below indexes K and V alike");

    // Lanes walk a K row with stride TPH, never in contiguous DPT-wide blocks:
    // at 4 bytes per half2 a block mapping puts lane l at bank (l*16)%32, i.e.
    // all TPH lanes of a warp on two banks -- an 8-way conflict on every K
    // access, in both the score and the accumulate loop. Striding puts them on
    // TPH consecutive banks instead. (Same bug fixed in flash_attn_sparse.comp;
    // there it was worth 3.2x.) Staging below is unaffected: 16-byte shared
    // writes are phased 8 threads at a time, which is already conflict-free.

    const int t  = blockIdx.x;             // query token
    const int hc = blockIdx.y;             // head chunk
    const int s  = blockIdx.z;             // stream

    const int tid  = threadIdx.x;
    const int hl   = tid / TPH;            // head within chunk
    const int lane = tid % TPH;            // slice within head
    const int h    = hc*HCHUNK + hl;       // global head
    const bool h_ok = h < p.n_head;

    __shared__ half2 Kt[TILE][DK/2];       // staged K/V rows (V = DV-prefix)
    __shared__ float Pt[HCHUNK][TILE];     // per-tile exp'd scores
    __shared__ unsigned char row_ok[TILE]; // gathered-row validity

    // Q slice for this (head, lane) in registers, pre-scaled
    float Qr[DPT];
    if (h_ok) {
        const float * q_ptr = (const float *) (p.q + t*p.nbq1 + h*p.nbq2 + s*p.nbq3);
#pragma unroll
        for (int i = 0; i < H2PT; ++i) {
            const int d0 = (i*TPH + lane)*2;
            Qr[2*i + 0] = q_ptr[d0 + 0]*p.scale;
            Qr[2*i + 1] = q_ptr[d0 + 1]*p.scale;
        }
    } else {
#pragma unroll
        for (int i = 0; i < DPT; ++i) {
            Qr[i] = 0.0f;
        }
    }

    float Or[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        Or[i] = 0.0f;
    }

    // Running softmax state. All TPH lanes of a head derive these from the same
    // cross-lane reductions, so they stay identical and live in registers.
    float M_run   = FATTN_SPARSE_M_INIT;
    float L_run   = 0.0f;
    float m_scale = 1.0f;

    const half  * mask_row = p.mask ? (const half *) (p.mask + t*p.nbm1 + s*p.nbm3) : nullptr;
    const int   * idx_row  = (const int *) (p.idx + t*p.nbi1 + s*p.nbi3);
    const int     n_vis    = *(const int *) (p.nvis + t*p.nbn0 + s*p.nbn3);

    const int n_tiles_dense  = (p.ne_dense + TILE - 1)/TILE;
    const int n_tiles_gather = (p.n_idx    + TILE - 1)/TILE;

    for (int tile = 0; tile < n_tiles_dense + n_tiles_gather; ++tile) {
        const bool gather = tile >= n_tiles_dense;
        const int  base   = gather ? (tile - n_tiles_dense)*TILE : tile*TILE;
        const int  n_rows = gather ? p.n_idx - base : p.ne_dense - base;

        // stage TILE rows into shared memory: 8 threads per row, int4 chunks
        {
            constexpr int TPR    = FATTN_SPARSE_NTH/TILE;     // threads per row
            constexpr int CHUNKS = DK*2/16;                   // int4 per row
            const int r  = tid / TPR;
            const int c0 = tid % TPR;

            bool ok = r < n_rows;
            const char * src = nullptr;
            if (ok) {
                if (gather) {
                    const int cell = idx_row[base + r];
                    ok = cell >= 0 && cell < n_vis;
                    if (ok) {
                        src = p.kc + (size_t) cell*p.nbc1 + s*p.nbc3;
                    }
                } else {
                    src = p.k + (size_t) (base + r)*p.nbk1 + s*p.nbk3;
                }
            }
            // one leader per row writes the flag unconditionally, so no
            // zero-init pass (and no extra __syncthreads) is needed
            if (c0 == 0) {
                row_ok[r] = ok ? 1 : 0;
            }
            if (ok) {
                const int4 * src4 = (const int4 *) src;
                int4 * dst4 = (int4 *) &Kt[r][0];
#pragma unroll
                for (int c = c0; c < CHUNKS; c += TPR) {
                    dst4[c] = src4[c];
                }
            }
        }
        __syncthreads();

        // scores: each head's TPH threads cooperate on the DK-dim dot product
        {
            for (int j = 0; j < TILE; ++j) {
                float sum = 0.0f;
                if (h_ok && row_ok[j]) {
                    const half2 * krow = &Kt[j][0];
#pragma unroll
                    for (int i = 0; i < H2PT; ++i) {
                        const float2 kv2 = __half22float2(krow[i*TPH + lane]);
                        sum += Qr[2*i + 0]*kv2.x + Qr[2*i + 1]*kv2.y;
                    }
                }
                // reduce across the TPH lanes of this head (contiguous threads)
#pragma unroll
                for (int off = TPH/2; off > 0; off >>= 1) {
                    sum += __shfl_xor_sync(0xffffffff, sum, off, TPH);
                }
                if (lane == 0) {
                    float sc = sum;
                    if (!row_ok[j]) {
                        sc = -INFINITY;
                    } else if (!gather && mask_row) {
                        // -inf mask entries flow through: exp(-inf - M) == 0 and
                        // fmaxf against M_INIT keeps M finite, so no NaN paths
                        sc += __half2float(mask_row[base + j]);
                    }
                    Pt[hl][j] = sc;
                }
            }
        }
        __syncthreads();

        // per-head online softmax update — the TPH lanes of a head split the
        // TILE rows, so every thread works (doing this on tid < HCHUNK left
        // 15/16 of the block stalled at the following barrier)
        {
            float tmax = FATTN_SPARSE_M_INIT;
#pragma unroll
            for (int j = lane; j < TILE; j += TPH) {
                tmax = fmaxf(tmax, Pt[hl][j]);
            }
#pragma unroll
            for (int off = TPH/2; off > 0; off >>= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffff, tmax, off, TPH));
            }

            const float Mnew = fmaxf(M_run, tmax);
            const float ms   = expf(M_run - Mnew);

            float l_add = 0.0f;
#pragma unroll
            for (int j = lane; j < TILE; j += TPH) {
                const float pv = expf(Pt[hl][j] - Mnew);
                Pt[hl][j] = pv;
                l_add += pv;
            }
#pragma unroll
            for (int off = TPH/2; off > 0; off >>= 1) {
                l_add += __shfl_xor_sync(0xffffffff, l_add, off, TPH);
            }

            M_run   = Mnew;
            L_run   = L_run*ms + l_add;
            m_scale = ms;
        }
        __syncthreads();

        // accumulate O — each thread owns a VPT-dim slice of one head
        {
            const float ms = m_scale;
            if (ms != 1.0f) {
#pragma unroll
                for (int i = 0; i < VPT; ++i) {
                    Or[i] *= ms;
                }
            }
            for (int j = 0; j < TILE; ++j) {
                const float pv = Pt[hl][j];
                if (pv == 0.0f) {
                    continue;
                }
                const half2 * vrow = &Kt[j][0]; // V = DV-prefix of the staged row
#pragma unroll
                for (int i = 0; i < H2PT; ++i) {
                    const float2 vv = __half22float2(vrow[i*TPH + lane]);
                    Or[2*i + 0] += pv*vv.x;
                    Or[2*i + 1] += pv*vv.y;
                }
            }
        }
        __syncthreads();
    }

    if (!h_ok) {
        return;
    }

    // sinks — every lane of the head already holds the same running state
    float ms_f = 1.0f;
    if (p.sinks) {
        const float sk   = ((const float *) p.sinks)[h];
        const float Mnew = fmaxf(M_run, sk);
        ms_f  = expf(M_run - Mnew);
        L_run = L_run*ms_f + expf(sk - Mnew);
    }

    const float li = L_run == 0.0f ? 0.0f : 1.0f/L_run;

    // dst is contiguous [DV, H, T, ns]
    float * out = p.dst + (size_t) DV*(h + (size_t) p.n_head*(t + (size_t) gridDim.x*s));
#pragma unroll
    for (int i = 0; i < H2PT; ++i) {
        const int d0 = (i*TPH + lane)*2;
        out[d0 + 0] = Or[2*i + 0]*ms_f*li;
        out[d0 + 1] = Or[2*i + 1]*ms_f*li;
    }
}

bool ggml_cuda_fattn_sparse_is_sparse_node(const ggml_tensor * dst) {
    return ggml_flash_attn_ext_is_sparse(dst);
}

bool ggml_cuda_fattn_sparse_supported(const ggml_tensor * dst) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * kc   = dst->src[5];
    const ggml_tensor * idx  = dst->src[6];
    const ggml_tensor * nvis = dst->src[7];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || kc->type != GGML_TYPE_F16) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (idx->type != GGML_TYPE_I32 || nvis->type != GGML_TYPE_I32) {
        return false;
    }
    if (k->ne[0] != 512 || v->ne[0] != 512 || kc->ne[0] != 512) {
        return false; // v1: DK == DV == 512 (DSV4)
    }
    if (k->ne[2] != 1 || v->ne[2] != 1 || kc->ne[2] != 1) {
        return false; // single KV head (MQA)
    }
    // the kernel stages one row per KV position and reads V as its DV-prefix,
    // so dense V must alias dense K (true for MLA/DSV4 where V is a K view)
    if (v->data != k->data || v->nb[1] != k->nb[1]) {
        return false;
    }
    // rows must be 16-byte aligned for int4 staging
    if (k->nb[1] % 16 != 0 || v->nb[1] % 16 != 0 || kc->nb[1] % 16 != 0) {
        return false;
    }
    return true;
}

void ggml_cuda_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * kc    = dst->src[5];
    const ggml_tensor * idx   = dst->src[6];
    const ggml_tensor * nvis  = dst->src[7];

    GGML_ASSERT(ggml_cuda_fattn_sparse_supported(dst));
    GGML_ASSERT(q->ne[0] == 512);

    fattn_sparse_params p = {};
    p.q     = (const char *) q->data;
    p.k     = (const char *) k->data;
    p.v     = (const char *) v->data;
    p.mask  = mask  ? (const char *) mask->data  : nullptr;
    p.sinks = sinks ? (const char *) sinks->data : nullptr;
    p.kc    = (const char *) kc->data;
    p.idx   = (const char *) idx->data;
    p.nvis  = (const char *) nvis->data;
    p.dst   = (float *) dst->data;

    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));

    p.ne_dense = (int) k->ne[1];
    p.n_idx    = (int) idx->ne[0];
    p.n_head   = (int) q->ne[2];

    p.nbq1 = q->nb[1];    p.nbq2 = q->nb[2];    p.nbq3 = q->nb[3];
    p.nbk1 = k->nb[1];    p.nbk3 = k->nb[3];
    p.nbv1 = v->nb[1];    p.nbv3 = v->nb[3];
    p.nbm1 = mask ? mask->nb[1] : 0;
    p.nbm3 = mask ? mask->nb[3] : 0;
    p.nbc1 = kc->nb[1];   p.nbc3 = kc->nb[3];
    p.nbi1 = idx->nb[1];  p.nbi3 = idx->nb[3];
    p.nbn0 = nvis->nb[0]; p.nbn3 = nvis->nb[3];

    const int T  = (int) q->ne[1];
    const int H  = (int) q->ne[2];
    const int ns = (int) q->ne[3];

    const dim3 grid(T, (H + FATTN_SPARSE_HCHUNK - 1)/FATTN_SPARSE_HCHUNK, ns);
    const dim3 block(FATTN_SPARSE_NTH, 1, 1);

    cudaStream_t stream = ctx.stream();
    flash_attn_sparse_f16<512, 512><<<grid, block, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}
