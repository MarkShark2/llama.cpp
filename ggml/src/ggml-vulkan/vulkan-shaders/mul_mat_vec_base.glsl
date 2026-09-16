#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#ifdef MUL_MAT_ID
#define EXPERT_COUNT 8
#endif

#include "mul_mat_vec_iface.glsl"

layout (push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;

#ifdef MUL_MAT_ID
    uint nei0;
    uint ne11;
    uint expert_i1;
    uint nbi1;
#else
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
#endif
} p;

layout (constant_id = 0) const uint BLOCK_SIZE = 32;
layout (constant_id = 1) const uint NUM_ROWS = 1;
layout (constant_id = 2) const uint NUM_COLS = 1;

#ifdef MUL_MAT_ID
uint expert_id;
#endif

// per-column B and D offsets, filled by get_offsets
#define MMV_MAX_COLS 8
uint b_off[MMV_MAX_COLS];
uint d_off[MMV_MAX_COLS];

#ifdef MUL_MAT_ID_GROUPED
// [fork] rows grouped by expert: the count_experts pre-pass (hoisted layout)
// holds per-expert counts, offsets and packed (token << 16 | slot) row ids;
// workgroup y is the expert and each chunk takes NUM_COLS of its rows.
// p.expert_i1 carries the expert count. Columns past the expert's last row
// compute on row 0's B and are not written.
layout (binding = 6) readonly buffer MAP {uint data_map[];};
uint mmv_chunk = 0;
uint mmv_ncols = 0;
uint mmv_pair[MMV_MAX_COLS];
#define MMV_COL_ACTIVE(j) ((j) < mmv_ncols)
#define MMV_FUSE_IDX(j) mmv_pair[j]
#define MMV_COMPUTE(first_row, num_rows) { \
    const uint mmv_count = data_map[gl_WorkGroupID.y]; \
    for (mmv_chunk = 0; mmv_chunk * NUM_COLS < mmv_count; ++mmv_chunk) { \
        if (mmv_chunk > 0) { barrier(); } \
        compute_outputs(first_row, num_rows); \
    } }
#else
#define MMV_COL_ACTIVE(j) true
#define MMV_FUSE_IDX(j) gl_GlobalInvocationID.y
#define MMV_COMPUTE(first_row, num_rows) compute_outputs(first_row, num_rows)
#endif

void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {
#ifdef MUL_MAT_ID_GROUPED
    expert_id = gl_WorkGroupID.y;
    const uint n_as  = p.expert_i1;
    const uint count = data_map[expert_id];
    const uint start = data_map[n_as + expert_id] + mmv_chunk * NUM_COLS;
    mmv_ncols = min(NUM_COLS, count - min(count, mmv_chunk * NUM_COLS));
    a_offset = expert_id * (p.batch_stride_a / QUANT_K);
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        const uint packed = data_map[2 * n_as + 1 + start + (j < mmv_ncols ? j : 0)];
        const uint t = packed >> 16;
        const uint s = packed & 0xffffu;
        mmv_pair[j] = t * p.nei0 + s;
        b_off[j] = (s % p.ne11) * p.stride_b + t * p.batch_stride_b;
        d_off[j] = s * p.stride_d + t * p.batch_stride_d;
    }
    b_offset = b_off[0];
    d_offset = d_off[0];
    return;
#endif
#ifdef MUL_MAT_ID
    const uint expert_i0 = gl_WorkGroupID.y;
#else
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;
#endif

#ifndef MUL_MAT_ID
    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }
#else
    expert_id = data_ids[expert_i0 + p.expert_i1 * p.nbi1];
#endif

    a_offset =
#ifdef MUL_MAT_ID
            expert_id * (p.batch_stride_a / QUANT_K);
#else
            batch_idx_a * (p.batch_stride_a / QUANT_K);
#endif
    b_offset =
#ifdef MUL_MAT_ID
            (expert_i0 % p.ne11) * p.stride_b + p.expert_i1 * p.batch_stride_b;
#else
            batch_idx * p.batch_stride_b;
#endif
    d_offset =
#ifdef MUL_MAT_ID
            expert_i0 * p.stride_d + p.expert_i1 * p.batch_stride_d;
#else
            batch_idx * p.batch_stride_d;
#endif
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        b_off[j] = j*p.batch_stride_b + b_offset;
        d_off[j] = j*p.batch_stride_d + d_offset;
    }
}

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[MMV_FUSE_IDX(j)]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[MMV_FUSE_IDX(j)]);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                if (MMV_COL_ACTIVE(j)) {
                    data_d[d_off[j] + first_row + n] = D_TYPE(temp[j][n]);
                }
            }
        }
    }
}
#else
shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];

void reduce_result(FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    // subgroupAdd is probably faster on devices that support it,
    // particularly when the workgroup has more than one subgroup
#if USE_SUBGROUP_ADD
    // sum up partial sums within a subgroup
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    // Go through shared memory to sum partials across subgroups
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    temp[j][n] += tmpsh[j][n][s];
                }
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[MMV_FUSE_IDX(j)]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[MMV_FUSE_IDX(j)]);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                if (MMV_COL_ACTIVE(j)) {
                    data_d[d_off[j] + first_row + n] = D_TYPE(temp[j][n]);
                }
            }
        }
    }
#else
    // sum up partial sums and write back result
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse0[MMV_FUSE_IDX(j)]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse1[MMV_FUSE_IDX(j)]);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
#endif
                if (MMV_COL_ACTIVE(j)) {
                    data_d[d_off[j] + first_row + n] = D_TYPE(tmpsh[j][n][0]);
                }
            }
        }
    }
#endif
}
#endif
