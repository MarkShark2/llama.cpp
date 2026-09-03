#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <future>
#include <algorithm>
#include <thread>
#include <ctime>
#include <cstdarg>
#include <cstdlib>
#ifndef _WIN32
#  include <fcntl.h>
#  include <unistd.h>
#else
#  include <process.h>
#endif

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

// defined next to the cache chunk size, used by set_tensor further up
static bool rpc_cache_write_file(const fs::path & path, const void * data, size_t size);

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_GRAPH_FORGET,
    // bf16 wire compression for f32 activations (GGML_RPC_WIRE_BF16=1):
    // same semantics as SET_TENSOR / GET_TENSOR but the wire payload is bf16
    // (half the bytes); the server expands/truncates at the buffer edge.
    RPC_CMD_SET_TENSOR_BF16,
    RPC_CMD_GET_TENSOR_BF16,
    // full-duplex transfer lanes (GGML_RPC_FULL_DUPLEX=1) [fork]:
    // SESSION_INFO returns the id of the server session so extra "lane"
    // connections can attach to it with LANE_ATTACH; LANE_FENCE orders the
    // main command stream after lane traffic (see the client lane section).
    RPC_CMD_SESSION_INFO,
    RPC_CMD_LANE_ATTACH,
    RPC_CMD_LANE_FENCE,
    // direct remote->remote activation transfer (GGML_RPC_PEER=1) [fork]:
    // PEER_OPEN tells a server to attach a peer lane to another server's
    // session; PUSH_TENSOR then makes it read one of its own tensors and ship
    // it straight down that lane, so a stage boundary never touches the client.
    RPC_CMD_PEER_OPEN,
    RPC_CMD_PUSH_TENSOR,
    // imatrix collection without dragging the activations home [fork]:
    // llama-imatrix only ever wants sum(x^2) per column (per expert, for
    // MUL_MAT_ID) of a matmul's src1. Reading that tensor back over the wire
    // costs ~10 GB per 512-token chunk on a 27-layer RPC split; the server can
    // do the reduction in its own RAM and answer with the accumulator instead,
    // which is token-count independent.
    RPC_CMD_IMATRIX_SQSUM,
    // fleet hibernation [fork]:
    // DETACH parks this session's buffers on the server and answers with a
    // token; the client then closes every connection so the host can suspend to
    // disk. RESUME re-adopts the parked buffers on a fresh connection. Without
    // it a disconnect destroys the session -- and with it the remote KV cache,
    // which is the one thing a hibernation cycle must not have to rebuild.
    RPC_CMD_SESSION_DETACH,
    RPC_CMD_SESSION_RESUME,
    RPC_CMD_COUNT,
};

// minimum server RPC_PROTO_PATCH_VERSION that understands the lane commands
#define GGML_RPC_FDX_MIN_PATCH 1
// ...and the peer (remote->remote) commands
#define GGML_RPC_PEER_MIN_PATCH 2
// ...and the server-side imatrix reduction
#define GGML_RPC_IMAT_MIN_PATCH 3
// ...and detach/resume of a parked session (fleet hibernation)
#define GGML_RPC_HIBERNATE_MIN_PATCH 4

enum rpc_lane_id : uint8_t {
    RPC_LANE_SET = 0,   // client -> server bulk uploads (fire-and-forget)
    RPC_LANE_GET = 1,   // client <- server bulk reads (request/response)
    RPC_LANE_PEER = 2,  // another server -> server bulk uploads (fire-and-forget)
};

// endpoint strings ("host:port") as carried on the wire
#define RPC_ENDPOINT_MAX 128

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

struct rpc_msg_graph_forget_req {
    uint32_t device;
    uint64_t uid;
};

struct rpc_msg_session_info_rsp {
    uint64_t session_id;
};

struct rpc_msg_lane_attach_req {
    uint64_t session_id;
    uint8_t  lane;      // rpc_lane_id
};

struct rpc_msg_lane_attach_rsp {
    uint8_t ok;
};

// main-lane barrier: wait until the lanes have fully processed the first
// wait_set / wait_get commands submitted on them
struct rpc_msg_session_detach_rsp {
    uint64_t token;      // 0 = the server refused to park
    uint64_t n_buffers;
};

struct rpc_msg_session_resume_req {
    uint64_t token;
};

struct rpc_msg_session_resume_rsp {
    uint32_t ok;
    uint32_t padding;
    uint64_t n_buffers;
};

struct rpc_msg_lane_fence_req {
    uint64_t wait_set;
    uint64_t wait_get;
};

// ask a server to open a peer lane into another server's session
struct rpc_msg_peer_open_req {
    uint64_t session_id;                  // the DESTINATION server's session id
    char     endpoint[RPC_ENDPOINT_MAX];  // "host:port" of the destination
};

struct rpc_msg_peer_open_rsp {
    uint8_t ok;
};

// read `src` locally and ship it to `endpoint` as a SET_TENSOR for `dst`.
// The dst tensor is serialized by the *client* and relayed verbatim, so the
// source server never has to reason about the destination's address space.
// wait_main / wait_get are the destination's lane fence targets, chosen by the
// client exactly as if it were sending this SET on the destination's SET lane.
struct rpc_msg_push_tensor_req {
    char       endpoint[RPC_ENDPOINT_MAX];
    rpc_tensor src;
    rpc_tensor dst;
    uint64_t   src_offset;
    uint64_t   dst_offset;
    uint64_t   size;        // bytes to read from src (pre-bf16 size)
    uint64_t   wait_main;   // destination-side fence targets
    uint64_t   wait_get;
    uint8_t    bf16;        // truncate to bf16 on the peer wire
};

struct rpc_msg_push_tensor_rsp {
    uint8_t ok;
};

// [fork] server-side imatrix reduction. src1 is the activation the client
// would otherwise GET in full; ids (MUL_MAT_ID only) says which expert each
// (n_expert_used, token) slot was routed to. The reply is
//   uint64 counts[n_mat]  followed by  float sums[src1.ne[0] * n_mat]
// and its size depends on n_mat and ne[0] only, never on the token count.
struct rpc_msg_imatrix_sqsum_req {
    rpc_tensor src1;
    rpc_tensor ids;       // ignored unless has_ids
    uint32_t   n_mat;     // src0->ne[2] for MUL_MAT_ID, src0->ne[2]*ne[3] otherwise
    uint32_t   has_ids;
    uint32_t   src0_ne2;  // dense path: mat_id = (i3 % src0_ne3)*src0_ne2 + (i2 % src0_ne2)
    uint32_t   src0_ne3;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

// [fork, PipeDec] number of deserialized graphs the remote server keeps per
// device. Multiple persistent scheds (stage-2 body lanes + the main sched)
// interleave stable-uid graphs on the same device, so a single-slot cache
// would never hit; the client mirrors the server's set and evicts explicitly
// with GRAPH_FORGET so a RECOMPUTE can never miss.
#define RPC_GRAPH_CACHE_SLOTS 32

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
    // uids of graphs the server holds deserialized for this device
    // (client-side LRU view, front = most recently used)
    std::vector<uint64_t> known_graph_uids;
    // [fork] struct-fingerprint -> content uid, see rpc_graph_quick_fp()
    std::unordered_map<uint64_t, uint64_t> quick_uids;
    // [fork] split-instance uid (ggml_cgraph::uid, assigned fresh by every
    // ggml_backend_sched_split_graph) -> content uid, see graph_compute
    std::unordered_map<uint64_t, uint64_t> split_uids;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
};

// note: deliberately holds no socket. A detach has to actually close the
// connection so the remote host can suspend, and a buffer context that pinned a
// shared_ptr<socket_t> would keep the fd open for the lifetime of the model.
// The socket is looked up per call instead - a mutex and a hash lookup against
// a network round trip.
struct ggml_backend_rpc_buffer_context {
    std::string endpoint;
    void * base_ptr;
    uint64_t remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    if (!sock->send_data(msg, msg_size)) {
        return false;
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// per-command wire accounting (client-side), enabled with GGML_RPC_CMD_STATS=1;
// printed every ~5s from whichever thread sends next
static void rpc_cmd_stats_add(enum rpc_cmd cmd, size_t bytes) {
    static const bool enabled = []{
        const char * e = std::getenv("GGML_RPC_CMD_STATS");
        return e && atoi(e) != 0;
    }();
    if (!enabled) {
        return;
    }
    static std::atomic<uint64_t> counts[RPC_CMD_COUNT] = {};
    static std::atomic<uint64_t> sizes[RPC_CMD_COUNT] = {};
    static std::atomic<int64_t> t_print{0};
    counts[cmd] += 1;
    sizes[cmd] += bytes;
    const int64_t now = ggml_time_us();
    int64_t prev = t_print.load();
    if (now - prev > 5*1000*1000 && t_print.compare_exchange_strong(prev, now)) {
        char buf[1024];
        size_t off = 0;
        for (int i = 0; i < RPC_CMD_COUNT && off < sizeof(buf) - 64; i++) {
            const uint64_t n = counts[i].exchange(0);
            const uint64_t s = sizes[i].exchange(0);
            if (n == 0) {
                continue;
            }
            off += (size_t) snprintf(buf + off, sizeof(buf) - off, " cmd%d n=%llu %.1fMB",
                    i, (unsigned long long) n, (double) s/1e6);
        }
        fprintf(stderr, "[rpc cmd stats]%s\n", buf);
        fflush(stderr);
    }
}

// [fork] per-endpoint wall accounting for the decode hot path, GGML_RPC_WIRE_TRACE=1.
//
// In the synchronous path (GGML_RPC_ASYNC=0) a stage's llama_decode is three
// distinct things on one ordered socket: upload the graph inputs (SET), submit
// the graph (fire-and-forget, no response), then read the output back (GET).
// Only the GET blocks, and it blocks across the whole remote compute -- so
// splitting set/submit/get separates "we are talking to the device" from "the
// device is working", which the per-stage timers upstack cannot distinguish.
// Bytes ride along so wire time can be divided out against the link rate.
struct rpc_wire_ep_stat {
    std::atomic<uint64_t> set_n{0}, set_us{0}, set_bytes{0};
    std::atomic<uint64_t> get_n{0}, get_us{0}, get_bytes{0};
    std::atomic<uint64_t> gc_n{0},  gc_us{0};
    // [fork] submit split: key_us = time to decide the uid (split cache /
    // fingerprint / content hash), full_n = graphs shipped in full
    std::atomic<uint64_t> gc_key_us{0}, gc_full_n{0}, gc_split_hit{0};
};

static std::mutex g_wire_stat_m;
static std::map<std::string, std::unique_ptr<rpc_wire_ep_stat>> g_wire_stats;

static int rpc_wire_trace_level() {
    static const int level = []{
        const char * e = std::getenv("GGML_RPC_WIRE_TRACE");
        return e ? atoi(e) : 0;
    }();
    return level;
}

static bool rpc_wire_trace_enabled() {
    return rpc_wire_trace_level() != 0;
}

// level 2: which tensors are being re-uploaded. A cache-hit graph should be
// staging almost nothing per eval, so the per-name breakdown is what says
// whether an input can be made cache-resident (the cache_k_rot pattern).
struct rpc_wire_name_stat {
    uint64_t n = 0, bytes = 0;
};
static std::map<std::string, rpc_wire_name_stat> g_wire_names;

static void rpc_wire_note_set(const ggml_tensor * tensor, size_t size) {
    if (rpc_wire_trace_level() < 2) {
        return;
    }
    char key[96];
    snprintf(key, sizeof(key), "%s [%s %lldx%lld]", tensor->name, ggml_type_name(tensor->type),
             (long long) tensor->ne[0], (long long) tensor->ne[1]);
    std::lock_guard<std::mutex> l(g_wire_stat_m);
    rpc_wire_name_stat & s = g_wire_names[key];
    s.n     += 1;
    s.bytes += size;
}

static rpc_wire_ep_stat * rpc_wire_stat(const std::string & endpoint) {
    std::lock_guard<std::mutex> l(g_wire_stat_m);
    auto it = g_wire_stats.find(endpoint);
    if (it != g_wire_stats.end()) {
        return it->second.get();
    }
    auto s = std::make_unique<rpc_wire_ep_stat>();
    rpc_wire_ep_stat * ptr = s.get();
    g_wire_stats[endpoint] = std::move(s);
    return ptr;
}

// printed every ~5s from whichever thread finishes an op next
static void rpc_wire_trace_tick() {
    static std::atomic<int64_t> t_print{0};
    const int64_t now = ggml_time_us();
    int64_t prev = t_print.load();
    if (now - prev <= 5*1000*1000 || !t_print.compare_exchange_strong(prev, now)) {
        return;
    }
    const double window_s = prev == 0 ? 0.0 : (now - prev)/1e6;
    std::lock_guard<std::mutex> l(g_wire_stat_m);
    for (auto & kv : g_wire_stats) {
        rpc_wire_ep_stat & s = *kv.second;
        const uint64_t set_n = s.set_n.exchange(0), set_us = s.set_us.exchange(0), set_b = s.set_bytes.exchange(0);
        const uint64_t get_n = s.get_n.exchange(0), get_us = s.get_us.exchange(0), get_b = s.get_bytes.exchange(0);
        const uint64_t gc_n  = s.gc_n.exchange(0),  gc_us  = s.gc_us.exchange(0);
        const uint64_t gc_key = s.gc_key_us.exchange(0), gc_full = s.gc_full_n.exchange(0), gc_hit = s.gc_split_hit.exchange(0);
        if (set_n == 0 && get_n == 0 && gc_n == 0) {
            continue;
        }
        fprintf(stderr, "[rpc wire] %-22s win=%.1fs | set n=%llu %.2fms %.1fMB | submit n=%llu %.3fms key=%.3fms split-hit=%llu full=%llu | get n=%llu %.2fms %.1fMB\n",
                kv.first.c_str(), window_s,
                (unsigned long long) set_n, set_n ? set_us/1e3/(double) set_n : 0.0, set_b/1e6,
                (unsigned long long) gc_n,  gc_n  ? gc_us /1e3/(double) gc_n  : 0.0,
                gc_n ? gc_key/1e3/(double) gc_n : 0.0,
                (unsigned long long) gc_hit, (unsigned long long) gc_full,
                (unsigned long long) get_n, get_n ? get_us/1e3/(double) get_n : 0.0, get_b/1e6);
    }
    if (rpc_wire_trace_level() >= 2 && !g_wire_names.empty()) {
        std::vector<std::pair<std::string, rpc_wire_name_stat>> rows(g_wire_names.begin(), g_wire_names.end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
            return a.second.bytes > b.second.bytes;
        });
        for (const auto & row : rows) {
            fprintf(stderr, "[rpc set-by-name] %-44s n=%llu %.1fMB (%.0f B each)\n",
                    row.first.c_str(), (unsigned long long) row.second.n, row.second.bytes/1e6,
                    row.second.n ? (double) row.second.bytes/(double) row.second.n : 0.0);
        }
        g_wire_names.clear();
    }
    fflush(stderr);
}

// [fork] Client-side write coalescing on the ordered command socket.
//
// A command frame is three send_data() calls -- cmd byte, size, payload -- so a
// DSV4 SPD stage graph, which stages ~23 inputs per eval (about 20 of them 1x1
// index scalars costing ~308 bytes each), spends ~70 blocking socket writes and
// the same number of reads in the server's command loop to move ~100 KB. The
// cost of a per-eval upload is dominated by the *command*, not the bytes:
// removing one 64 KiB constant was worth 2.3 ms of stage latency against 0.56 ms
// of link time, and 3.7 ms on a loopback stage that has no network at all.
//
// Frames that expect no response are appended to a per-socket buffer and leave
// as one write. Ordering is exact: bytes are appended in call order, and the
// buffer is flushed before anything reads from the socket. Graph submissions
// append and flush immediately -- they are fire-and-forget, so a pending one
// would leave the remote idle until some unrelated call happened along.
//
// Set GGML_RPC_SEND_COALESCE=0 to send frame-per-write again (the A/B control).
static constexpr size_t RPC_SEND_COALESCE_MAX_PAYLOAD = 256*1024;
static constexpr size_t RPC_SEND_COALESCE_FLUSH_AT    = 1024*1024;

struct rpc_send_queue {
    std::mutex m;
    std::vector<uint8_t> pending;
};

static std::mutex g_send_queue_map_m;
static std::unordered_map<socket_t *, std::unique_ptr<rpc_send_queue>> g_send_queues;

// This sits on the path of every command, so it must not take a process-global
// lock in the common case: a stage thread talks to one endpoint, so a one-entry
// thread-local cache hits essentially always. Entries are never erased, and the
// map is node-based, so the raw pointer stays valid for the process lifetime.
static rpc_send_queue * rpc_send_queue_for(socket_t * key) {
    static thread_local socket_t       * last_key = nullptr;
    static thread_local rpc_send_queue * last_q   = nullptr;
    if (key == last_key && last_q != nullptr) {
        return last_q;
    }
    std::lock_guard<std::mutex> l(g_send_queue_map_m);
    auto & slot = g_send_queues[key];
    if (slot == nullptr) {
        slot = std::make_unique<rpc_send_queue>();
    }
    last_key = key;
    last_q   = slot.get();
    return last_q;
}

static bool rpc_send_coalesce_enabled() {
    static const bool enabled = []{
        const char * e = std::getenv("GGML_RPC_SEND_COALESCE");
        return e == nullptr || atoi(e) != 0;
    }();
    return enabled;
}

// caller holds q.m
static bool rpc_send_flush_locked(const socket_ptr & sock, rpc_send_queue & q) {
    if (q.pending.empty()) {
        return true;
    }
    const bool ok = sock->send_data(q.pending.data(), q.pending.size());
    q.pending.clear();
    // message boundary: the RDMA transport posts its trailing partial frame
    // only on flush() (no-op on TCP)
    return ok && sock->flush();
}

// caller holds q.m -- appends one frame and sends the buffer when it must not
// be left pending
static bool rpc_send_frame_locked(
        const socket_ptr & sock, rpc_send_queue & q,
        enum rpc_cmd cmd, const void * input, size_t input_size, bool flush) {
    const uint8_t cmd_byte = cmd;
    const uint64_t size64  = input_size;
    const size_t   base    = q.pending.size();
    q.pending.resize(base + sizeof(cmd_byte) + sizeof(size64) + input_size);
    uint8_t * dst = q.pending.data() + base;
    memcpy(dst, &cmd_byte, sizeof(cmd_byte));
    dst += sizeof(cmd_byte);
    memcpy(dst, &size64, sizeof(size64));
    dst += sizeof(size64);
    if (input_size > 0) {
        memcpy(dst, input, input_size);
    }
    if (flush || q.pending.size() >= RPC_SEND_COALESCE_FLUSH_AT) {
        return rpc_send_flush_locked(sock, q);
    }
    return true;
}

// a frame that must go out now: everything the remote could be left waiting on,
// plus anything too large to be worth staging through the buffer
static bool rpc_send_must_flush(enum rpc_cmd cmd, size_t input_size) {
    return !rpc_send_coalesce_enabled() ||
           input_size > RPC_SEND_COALESCE_MAX_PAYLOAD ||
           cmd == RPC_CMD_GRAPH_COMPUTE ||
           cmd == RPC_CMD_GRAPH_RECOMPUTE;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    if (sock == nullptr) {
        return false;
    }
    rpc_cmd_stats_add(cmd, input_size);
    rpc_send_queue * q = rpc_send_queue_for(sock.get());
    std::lock_guard<std::mutex> l(q->m);
    return rpc_send_frame_locked(sock, *q, cmd, input, input_size,
            rpc_send_must_flush(cmd, input_size));
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (sock == nullptr) {
        return false;
    }
    rpc_cmd_stats_add(cmd, input_size);
    // the send and the matching read are one transaction: nothing else may put
    // bytes on this socket between them
    rpc_send_queue * q = rpc_send_queue_for(sock.get());
    std::lock_guard<std::mutex> l(q->m);
    if (!rpc_send_frame_locked(sock, *q, cmd, input, input_size, /*flush =*/ true)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// When enabled, large tensor uploads are hashed and offered to the server's
// local tensor cache (SET_TENSOR_HASH). The server only writes new cache
// entries for uploads that went through this path, so this flag effectively
// controls per-client whether the model gets cached on the server.
static std::atomic<bool> g_rpc_client_cache{false};

void ggml_backend_rpc_set_client_cache(bool enabled) {
    g_rpc_client_cache.store(enabled, std::memory_order_relaxed);
}

bool ggml_backend_rpc_get_client_cache(void) {
    return g_rpc_client_cache.load(std::memory_order_relaxed);
}

size_t ggml_backend_rpc_cache_threshold(void) {
    return HASH_THRESHOLD;
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, uint8_t * out_patch = nullptr) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    if (!status) {
        // do not abort here: the connection may have been accepted and dropped
        // (e.g. server busy with another client) - let the caller retry
        GGML_LOG_WARN("[%s] HELLO handshake failed\n", __func__);
        return false;
    }

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    if (out_patch != nullptr) {
        *out_patch = response.patch;
    }
    sock->update_caps(response.conn_caps);
    return true;
}

// server patch version per endpoint, recorded when the main socket handshakes;
// gates the full-duplex lane attach so new clients keep working on old daemons
static std::mutex g_server_patch_mutex;
static std::unordered_map<std::string, uint8_t> g_server_patch;

static uint8_t rpc_server_patch(const std::string & endpoint) {
    std::lock_guard<std::mutex> lock(g_server_patch_mutex);
    auto it = g_server_patch.find(endpoint);
    return it != g_server_patch.end() ? it->second : 0;
}

// ---------------------------------------------------------------------------
// Endpoint socket table
//
// A strong reference to every endpoint socket is held for the lifetime of the
// process: the rpc-server serves one client at a time, so reconnecting per
// operation (what a weak_ptr cache degrades to whenever no buffer holds a
// strong ref) floods the server with one-shot connections and starves every
// reconnect attempt while the server is busy with a long request.
//
// The table lives at file scope rather than inside get_socket() because fleet
// hibernation has to walk it: detaching means closing every one of these, and
// nothing may reopen one behind our back while the hosts are asleep.
// ---------------------------------------------------------------------------
static std::mutex g_sockets_m;
static std::unordered_map<std::string, std::shared_ptr<socket_t>> g_sockets;
// every endpoint this process has connected to, in first-contact order, so
// status reporting keeps a stable index even while nothing is connected
static std::vector<std::string> g_endpoints_seen;
// set between a detach and a successful reattach. get_socket() refuses to dial
// while it is set, so a stray buffer free cannot wake a suspended host - or
// worse, connect to one that is halfway through writing its suspend image.
static std::atomic<bool> g_rpc_detached{false};
static std::atomic<bool> g_rpc_session_lost{false};

static void rpc_note_endpoint(const std::string & endpoint) {
    if (std::find(g_endpoints_seen.begin(), g_endpoints_seen.end(), endpoint) == g_endpoints_seen.end()) {
        g_endpoints_seen.push_back(endpoint);
    }
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    std::lock_guard<std::mutex> lock(g_sockets_m);
    auto & sockets = g_sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        return it->second;
    }
    if (g_rpc_detached.load()) {
        return nullptr;
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    // the rpc-server handles one client at a time, so transient connect
    // failures are expected when several clients/probes hit the same
    // endpoint - retry with backoff before giving up
    constexpr int max_attempts = 5;
    std::shared_ptr<socket_t> sock;
    uint8_t server_patch = 0;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        sock = socket_t::connect(host.c_str(), port);
        if (sock != nullptr && negotiate_hello(sock, &server_patch)) {
            break;
        }
        sock = nullptr;
        if (attempt < max_attempts) {
            int delay_ms = 250 * attempt;
            GGML_LOG_WARN("[%s] connect to %s failed (attempt %d/%d), retrying in %d ms\n",
                          __func__, endpoint.c_str(), attempt, max_attempts, delay_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        } else {
            GGML_LOG_ERROR("[%s] connect to %s failed after %d attempts\n",
                           __func__, endpoint.c_str(), max_attempts);
        }
    }
    if (sock == nullptr) {
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> plock(g_server_patch_mutex);
        g_server_patch[endpoint] = server_patch;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
    rpc_note_endpoint(endpoint);
    return sock;
}

// ---------------------------------------------------------------------------
// Async RPC streams + events  [fork, PipeDec Phase 1]
//
// Opt-in via GGML_RPC_ASYNC=1. Each endpoint's persistent socket is already a
// strictly in-order command stream (the server serves one client, FIFO). We
// drive it from a dedicated worker thread so the scheduler thread can hand work
// to several endpoints without blocking on any single one - which is what lets
// ggml_backend_sched overlap the pipeline stages of a layer-split model and
// fill the decode "bubble". Client-only: the rpc-server is unchanged.
// ---------------------------------------------------------------------------

static bool rpc_async_enabled() {
    static const bool enabled = []{
        const char * e = std::getenv("GGML_RPC_ASYNC");
        return e && atoi(e) != 0;
    }();
    return enabled;
}

// one-shot data-ready gate used to hand a tensor payload between two streams
struct rpc_gate {
    std::mutex m;
    std::condition_variable cv;
    bool ready = false;
    void set()  { std::lock_guard<std::mutex> l(m); ready = true; cv.notify_all(); }
    void wait() { std::unique_lock<std::mutex> l(m); cv.wait(l, [&]{ return ready; }); }
};

// reusable event: monotonic "record" generation vs highest "completed" generation.
// A gen counter (rather than a bool) is robust to the scheduler reusing one event
// object across n_copies iterations while an older completion is still in flight.
struct rpc_event {
    std::mutex m;
    std::condition_variable cv;
    uint64_t last = 0;   // gen assigned by the most recent record
    uint64_t done = 0;   // highest gen the stream has completed
    uint64_t record() { std::lock_guard<std::mutex> l(m); return ++last; }
    uint64_t peek()   { std::lock_guard<std::mutex> l(m); return last; }
    void complete(uint64_t g) { std::lock_guard<std::mutex> l(m); if (g > done) { done = g; } cv.notify_all(); }
    void wait_for(uint64_t target) { std::unique_lock<std::mutex> l(m); cv.wait(l, [&]{ return done >= target; }); }
};

// per-endpoint worker thread that owns socket IO for that endpoint
struct rpc_stream {
    std::string endpoint;
    std::thread worker;
    std::mutex m;
    std::condition_variable cv;       // tasks available / stop
    std::condition_variable cv_done;  // completed == submitted
    std::deque<std::function<void()>> tasks;
    uint64_t submitted = 0;
    uint64_t completed = 0;
    // barrier bookkeeping: a synchronize() only needs the server round-trip ping
    // when work was enqueued after the last completed ping. dirty_seq = submitted
    // index of the newest task needing a barrier; barrier_seq = highest submitted
    // index known to be server-drained.
    uint64_t dirty_seq   = 0;
    uint64_t barrier_seq = 0;
    bool stop = false;

    explicit rpc_stream(std::string ep) : endpoint(std::move(ep)) {
        worker = std::thread([this]{ run(); });
    }

    ~rpc_stream() {
        {
            std::lock_guard<std::mutex> l(m);
            stop = true;
        }
        cv.notify_one();
        if (worker.joinable()) {
            worker.join();
        }
    }

    void enqueue(std::function<void()> fn) {
        std::lock_guard<std::mutex> l(m);
        tasks.push_back(std::move(fn));
        submitted++;
        dirty_seq = submitted;
        cv.notify_one();
    }

    // enqueue a barrier ping; returns the submitted index it covers. Does NOT
    // mark the stream dirty (the ping itself needs no later barrier).
    uint64_t enqueue_barrier(std::function<void()> fn) {
        std::lock_guard<std::mutex> l(m);
        tasks.push_back(std::move(fn));
        submitted++;
        cv.notify_one();
        return submitted;
    }

    bool needs_barrier() {
        std::lock_guard<std::mutex> l(m);
        return dirty_seq > barrier_seq;
    }

    void mark_barrier(uint64_t covered) {
        std::lock_guard<std::mutex> l(m);
        if (covered > barrier_seq) {
            barrier_seq = covered;
        }
    }

    // Run a response-bearing operation in FIFO order on this endpoint. The
    // caller remains synchronous, but only the stream worker touches the
    // socket while async RPC is enabled.
    bool call(std::function<bool()> fn) {
        if (std::this_thread::get_id() == worker.get_id()) {
            return fn();
        }
        auto result = std::make_shared<std::promise<bool>>();
        auto future = result->get_future();
        enqueue([fn = std::move(fn), result] {
            try {
                result->set_value(fn());
            } catch (...) {
                result->set_exception(std::current_exception());
            }
        });
        return future.get();
    }

    // block the calling (host) thread until every enqueued task has finished
    void drain() {
        std::unique_lock<std::mutex> l(m);
        cv_done.wait(l, [&]{ return completed == submitted; });
    }

    // [fork, chained decode] ordinal-scoped wait: block only until tasks
    // enqueued at snapshot time have finished, not the whole stream
    uint64_t submitted_seq() {
        std::lock_guard<std::mutex> l(m);
        return submitted;
    }

    void wait_completed(uint64_t target) {
        std::unique_lock<std::mutex> l(m);
        cv_done.wait(l, [&]{ return completed >= target; });
    }

    void run() {
        for (;;) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> l(m);
                cv.wait(l, [&]{ return stop || !tasks.empty(); });
                if (stop && tasks.empty()) {
                    return;
                }
                fn = std::move(tasks.front());
                tasks.pop_front();
            }
            fn();
            {
                std::lock_guard<std::mutex> l(m);
                completed++;
                cv_done.notify_all();
            }
        }
    }
};

static std::mutex g_streams_m;
static std::unordered_map<std::string, std::unique_ptr<rpc_stream>> g_streams;

static rpc_stream * get_stream(const std::string & endpoint) {
    auto & streams = g_streams;
    std::lock_guard<std::mutex> lock(g_streams_m);
    auto it = streams.find(endpoint);
    if (it != streams.end()) {
        return it->second.get();
    }
    auto s = std::make_unique<rpc_stream>(endpoint);
    rpc_stream * ptr = s.get();
    streams[endpoint] = std::move(s);
    return ptr;
}

// ---------------------------------------------------------------------------
// Full-duplex transfer lanes  [fork, pipeline-prefill Phase 2]
//
// With a single socket per endpoint, the daemon cannot drain the next ubatch's
// input SET (or send the previous ubatch's output GET response) while its
// blocking backend compute call owns the command loop - every pipeline stage
// pays wire-transfer time between computes (measured: ~50% idle per board on
// the 8-board DSV4 prefill). Opt-in via GGML_RPC_FULL_DUPLEX=1 (requires
// GGML_RPC_ASYNC=1 and a server with proto patch >= GGML_RPC_FDX_MIN_PATCH):
// two extra connections per endpoint carry the bulk tensor traffic -
//   SET lane: SET_TENSOR / SET_TENSOR_BF16, fire-and-forget
//   GET lane: GET_TENSOR / GET_TENSOR_BF16, request/response
// Correctness: the client counts commands per lane in submission order; every
// lane command carries the other two lanes' counts at its submission, and the
// main lane gets a LANE_FENCE with the lane counts before ordered commands.
// The server executes a command only once the other lanes have processed the
// counts it carries - this reconstructs the exact single-socket execution
// order, but the wire transfers (and the daemon-side deserialization) overlap
// with compute instead of serializing against it.
// ---------------------------------------------------------------------------

static bool rpc_fdx_enabled() {
    static const bool enabled = []{
        const char * e = std::getenv("GGML_RPC_FULL_DUPLEX");
        return e && atoi(e) != 0;
    }();
    return enabled;
}

struct rpc_ep_lanes {
    std::mutex m;
    // client-side wire-command counts per lane since HELLO, in submission order
    uint64_t main_enq = 0, set_enq = 0, get_enq = 0;
    // lane counts covered by the most recent LANE_FENCE enqueued on the main lane
    uint64_t fenced_set = 0, fenced_get = 0;
    // lane counts covered by the last completed synchronize barrier
    uint64_t barrier_set = 0, barrier_get = 0;
    int      state = 0;   // 0 = untried, 1 = active, -1 = unavailable
    uint64_t session_id = 0;
    socket_ptr   set_sock, get_sock;
    rpc_stream * set_stream = nullptr;
    rpc_stream * get_stream = nullptr;
};

static std::mutex g_lanes_m;
static std::unordered_map<std::string, std::unique_ptr<rpc_ep_lanes>> g_lanes;

static rpc_ep_lanes * get_ep_lanes(const std::string & endpoint) {
    auto & lanes = g_lanes;
    std::lock_guard<std::mutex> lock(g_lanes_m);
    auto it = lanes.find(endpoint);
    if (it != lanes.end()) {
        return it->second.get();
    }
    auto l = std::make_unique<rpc_ep_lanes>();
    rpc_ep_lanes * ptr = l.get();
    lanes[endpoint] = std::move(l);
    return ptr;
}

// lane wire framing: | cmd (1) | size (8) = 16 + payload | wait_a (8) | wait_b (8) | payload |
// wait_a is always the main-lane count; wait_b is the opposite lane's count.
static bool send_lane_cmd(const socket_ptr & sock, enum rpc_cmd cmd,
                          uint64_t wait_a, uint64_t wait_b, const void * data, size_t size) {
    rpc_cmd_stats_add(cmd, size);
    uint8_t cmd_byte = cmd;
    uint64_t total = 2*sizeof(uint64_t) + size;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&total, sizeof(total))) {
        return false;
    }
    if (!sock->send_data(&wait_a, sizeof(wait_a))) {
        return false;
    }
    if (!sock->send_data(&wait_b, sizeof(wait_b))) {
        return false;
    }
    return sock->send_data(data, size);
}

// Enqueue a main-lane task that sends n_cmds wire commands. When lane traffic
// advanced since the last fence, a LANE_FENCE is sent first so the server
// orders this command after every lane command submitted before it.
static void rpc_main_enqueue_counted(const std::string & endpoint, rpc_stream * st,
                                     uint32_t n_cmds, std::function<void()> fn) {
    rpc_ep_lanes * ep = get_ep_lanes(endpoint);
    std::lock_guard<std::mutex> l(ep->m);
    const bool fence = ep->state == 1 && (ep->set_enq != ep->fenced_set || ep->get_enq != ep->fenced_get);
    if (!fence) {
        ep->main_enq += n_cmds;
        st->enqueue(std::move(fn));
        return;
    }
    rpc_msg_lane_fence_req freq = { ep->set_enq, ep->get_enq };
    ep->fenced_set = ep->set_enq;
    ep->fenced_get = ep->get_enq;
    ep->main_enq += n_cmds + 1;
    st->enqueue([endpoint, freq, fn = std::move(fn)] {
        auto sock = get_socket(endpoint);
        if (sock != nullptr) {
            send_rpc_cmd(sock, RPC_CMD_LANE_FENCE, &freq, sizeof(freq));
        }
        fn();
    });
}

// Run a response-bearing main-lane operation in FIFO order, counted for the
// lane fences. Mirrors rpc_stream::call - a nested call from the stream worker
// itself runs inline at the socket's current position (counting it then only
// makes later fence targets conservative, never too early).
static bool rpc_main_call_counted(const std::string & endpoint, rpc_stream * st, std::function<bool()> fn) {
    rpc_ep_lanes * ep = get_ep_lanes(endpoint);
    if (std::this_thread::get_id() == st->worker.get_id()) {
        {
            std::lock_guard<std::mutex> l(ep->m);
            ep->main_enq++;
        }
        return fn();
    }
    auto result = std::make_shared<std::promise<bool>>();
    auto future = result->get_future();
    rpc_main_enqueue_counted(endpoint, st, 1, [fn = std::move(fn), result] {
        try {
            result->set_value(fn());
        } catch (...) {
            result->set_exception(std::current_exception());
        }
    });
    return future.get();
}

static socket_ptr rpc_lane_connect(const std::string & endpoint, uint64_t session_id, uint8_t lane) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    rpc_msg_lane_attach_req request = { session_id, lane };
    rpc_msg_lane_attach_rsp response = { 0 };
    if (!send_rpc_cmd(sock, RPC_CMD_LANE_ATTACH, &request, sizeof(request), &response, sizeof(response)) ||
        !response.ok) {
        return nullptr;
    }
    return sock;
}

// Bring up the transfer lanes for an endpoint (tried once); returns the lane
// state when active, nullptr when unavailable (callers fall back to the main
// lane - mixed routing stays correct because the fences only count commands).
static rpc_ep_lanes * rpc_lanes_get_active(const std::string & endpoint) {
    if (!rpc_async_enabled() || !rpc_fdx_enabled()) {
        return nullptr;
    }
    rpc_ep_lanes * ep = get_ep_lanes(endpoint);
    {
        std::lock_guard<std::mutex> l(ep->m);
        if (ep->state == 1) {
            return ep;
        }
        if (ep->state != 0) {
            return nullptr;
        }
        ep->state = -1;   // claim; flipped to 1 only if the whole setup succeeds
    }
    if (rpc_server_patch(endpoint) < GGML_RPC_FDX_MIN_PATCH) {
        GGML_LOG_WARN("[rpc fdx] %s: server too old for transfer lanes, staying single-socket\n", endpoint.c_str());
        return nullptr;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        return nullptr;
    }
    rpc_msg_session_info_rsp info = {};
    bool ok = rpc_main_call_counted(endpoint, get_stream(endpoint), [sock, &info] {
        return send_rpc_cmd(sock, RPC_CMD_SESSION_INFO, nullptr, 0, &info, sizeof(info));
    });
    if (!ok) {
        GGML_LOG_WARN("[rpc fdx] %s: SESSION_INFO failed, staying single-socket\n", endpoint.c_str());
        return nullptr;
    }
    socket_ptr set_sock = rpc_lane_connect(endpoint, info.session_id, RPC_LANE_SET);
    socket_ptr get_sock = rpc_lane_connect(endpoint, info.session_id, RPC_LANE_GET);
    if (set_sock == nullptr || get_sock == nullptr) {
        GGML_LOG_WARN("[rpc fdx] %s: lane attach failed, staying single-socket\n", endpoint.c_str());
        return nullptr;
    }
    {
        std::lock_guard<std::mutex> l(ep->m);
        ep->session_id = info.session_id;
        ep->set_sock   = set_sock;
        ep->get_sock   = get_sock;
        ep->set_stream = new rpc_stream(endpoint + "/set");
        ep->get_stream = new rpc_stream(endpoint + "/get");
        ep->state      = 1;
    }
    GGML_LOG_INFO("[rpc fdx] %s: transfer lanes active (session %" PRIu64 ")\n", endpoint.c_str(), info.session_id);
    return ep;
}

// ---------------------------------------------------------------------------
// Direct remote->remote transfer (GGML_RPC_PEER=1)  [fork]
//
// In the RPC star every stage boundary hairpins through the client: GET the
// tensor off the producer, SET it onto the consumer. Both legs cross the
// client's single NIC, which is what makes long-prompt prefill on a multi-node
// split transport-bound rather than compute-bound.
//
// With a peer route the producer ships the payload itself, over whatever link
// the two nodes share, and the client sends only a small PUSH_TENSOR. The
// ordering story is unchanged: the push is issued on the producer's GET lane
// (so it lands after that node's compute, exactly like the GET it replaces),
// and it carries the consumer's SET-lane fence targets, so the consumer's
// executor applies it in the same position the client's own SET would have
// taken. Both endpoints therefore need active lanes; without them there is no
// ordering domain to slot into and we stay on the hairpin.
// ---------------------------------------------------------------------------
static bool rpc_peer_enabled() {
    static const bool on = []() {
        const char * e = std::getenv("GGML_RPC_PEER");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    return on;
}

// GGML_RPC_PEER_MAP="<client endpoint>=<peer endpoint>,..." rewrites the
// address a producer dials for a destination. The client reaches the nodes on
// whatever network it shares with them, which is not necessarily the fastest
// link *between* them: here the boards sit on a 10 GbE fabric the Windows head
// is not even attached to, so without this the peer traffic would take the
// management LAN. Unmapped endpoints are dialed exactly as the client has them.
static const std::string & rpc_peer_addr(const std::string & endpoint) {
    static const std::unordered_map<std::string, std::string> map = []{
        std::unordered_map<std::string, std::string> m;
        const char * e = std::getenv("GGML_RPC_PEER_MAP");
        if (e == nullptr) {
            return m;
        }
        std::string spec(e);
        size_t pos = 0;
        while (pos < spec.size()) {
            size_t comma = spec.find(',', pos);
            if (comma == std::string::npos) {
                comma = spec.size();
            }
            const std::string entry = spec.substr(pos, comma - pos);
            const size_t eq = entry.find('=');
            if (eq != std::string::npos && eq > 0 && eq + 1 < entry.size()) {
                m[entry.substr(0, eq)] = entry.substr(eq + 1);
                GGML_LOG_INFO("[rpc peer] route %s via %s\n",
                              entry.substr(0, eq).c_str(), entry.substr(eq + 1).c_str());
            } else if (!entry.empty()) {
                GGML_LOG_WARN("[rpc peer] ignoring malformed GGML_RPC_PEER_MAP entry '%s'\n", entry.c_str());
            }
            pos = comma + 1;
        }
        return m;
    }();
    auto it = map.find(endpoint);
    return it == map.end() ? endpoint : it->second;
}

static std::mutex g_peer_route_m;
static std::unordered_map<std::string, bool> g_peer_routes;   // "src>dst" -> usable

// Ask `src_endpoint` to open a peer lane into `dst_endpoint`'s session. Tried
// once per ordered pair; a failure is remembered so the pair quietly keeps
// using the hairpin (routing is per-pair, so a fabric that only connects some
// of the nodes still gets the benefit on the pairs that do).
static bool rpc_peer_route_ready(const std::string & src_endpoint,
                                 const std::string & dst_endpoint,
                                 uint64_t dst_session_id) {
    const std::string key = src_endpoint + ">" + dst_endpoint;
    {
        std::lock_guard<std::mutex> l(g_peer_route_m);
        auto it = g_peer_routes.find(key);
        if (it != g_peer_routes.end()) {
            return it->second;
        }
    }
    const std::string & via = rpc_peer_addr(dst_endpoint);
    bool ok = false;
    if (via.size() >= RPC_ENDPOINT_MAX) {
        GGML_LOG_WARN("[rpc peer] endpoint '%s' too long to route\n", via.c_str());
    } else if (rpc_server_patch(src_endpoint) < GGML_RPC_PEER_MIN_PATCH ||
               rpc_server_patch(dst_endpoint) < GGML_RPC_PEER_MIN_PATCH) {
        GGML_LOG_WARN("[rpc peer] %s -> %s: server too old for peer transfer\n",
                      src_endpoint.c_str(), dst_endpoint.c_str());
    } else {
        rpc_msg_peer_open_req req = {};
        req.session_id = dst_session_id;
        memcpy(req.endpoint, via.c_str(), via.size());
        rpc_msg_peer_open_rsp rsp = { 0 };
        auto sock = get_socket(src_endpoint);
        if (sock != nullptr) {
            const bool sent = rpc_main_call_counted(src_endpoint, get_stream(src_endpoint),
                                                    [&sock, &req, &rsp] {
                return send_rpc_cmd(sock, RPC_CMD_PEER_OPEN, &req, sizeof(req), &rsp, sizeof(rsp));
            });
            ok = sent && rsp.ok != 0;
        }
        if (ok) {
            GGML_LOG_INFO("[rpc peer] %s -> %s: direct transfer\n",
                          src_endpoint.c_str(), dst_endpoint.c_str());
        } else {
            GGML_LOG_WARN("[rpc peer] %s -> %s: peer lane refused, using the client hairpin\n",
                          src_endpoint.c_str(), dst_endpoint.c_str());
        }
    }
    std::lock_guard<std::mutex> l(g_peer_route_m);
    g_peer_routes[key] = ok;
    return ok;
}

static bool send_rpc_cmd_ordered(
        const std::string & endpoint, socket_ptr sock, enum rpc_cmd cmd,
        const void * input, size_t input_size) {
    if (!rpc_async_enabled()) {
        return send_rpc_cmd(sock, cmd, input, input_size);
    }
    return rpc_main_call_counted(endpoint, get_stream(endpoint), [sock, cmd, input, input_size] {
        return send_rpc_cmd(sock, cmd, input, input_size);
    });
}

static bool send_rpc_cmd_ordered(
        const std::string & endpoint, socket_ptr sock, enum rpc_cmd cmd,
        const void * input, size_t input_size, void * output, size_t output_size) {
    if (!rpc_async_enabled()) {
        return send_rpc_cmd(sock, cmd, input, input_size, output, output_size);
    }
    return rpc_main_call_counted(endpoint, get_stream(endpoint), [sock, cmd, input, input_size, output, output_size] {
        return send_rpc_cmd(sock, cmd, input, input_size, output, output_size);
    });
}

// cheap round-trip that returns only after the server has drained all prior
// in-order commands on this socket (so a preceding fire-and-forget GRAPH_COMPUTE
// is known to have finished). Used as the async synchronize() barrier.
static void rpc_ping(const std::string & endpoint, uint32_t device) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        return;
    }
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto sock = get_socket(ctx->endpoint);
    if (sock == nullptr) {
        // detached for host hibernation, or the endpoint is gone. The far side
        // owns this buffer until its session is discarded, and aborting the
        // process on the teardown path helps nobody.
        GGML_LOG_DEBUG("[%s] %s is not connected; leaving the remote buffer to "
                       "its parked session\n", __func__, ctx->endpoint.c_str());
        delete ctx;
        return;
    }
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd_ordered(ctx->endpoint, sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        bool status = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

// [fork note] the fill is described, never transferred: DeepSeek-V4's DSA KV cache
// memsets a per-stream slice on every sequence clear (llama-kv-cache-dsv4.cpp), and
// the fork's old emulation shipped that whole slice as SET_TENSOR payload.
static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_memset_tensor_req request = {
        /* .tensor = */ serialize_tensor(tensor),
        /* .offset = */ offset,
        /* .size   = */ size,
        /* .value  = */ value,
    };
    bool status = send_rpc_cmd(get_socket(ctx->endpoint), RPC_CMD_MEMSET_TENSOR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static bool rpc_buffer_set_tensor_raw(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    {
        static const int stats_level = []{
            const char * e = std::getenv("GGML_RPC_CMD_STATS");
            return e ? atoi(e) : 0;
        }();
        if (stats_level >= 2 && size >= 32*1024) {
            fprintf(stderr, "[rpc set_tensor] %s type=%s size=%zu\n",
                    tensor->name, ggml_type_name(tensor->type), size);
        }
    }
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    if (rpc_wire_trace_enabled()) {
        rpc_wire_ep_stat * st = rpc_wire_stat(ctx->endpoint);
        rpc_wire_note_set(tensor, input.size());
        const int64_t t0 = ggml_time_us();
        bool ok = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_SET_TENSOR, input.data(), input.size());
        st->set_us    += (uint64_t) (ggml_time_us() - t0);
        st->set_bytes += input.size();
        st->set_n     += 1;
        rpc_wire_trace_tick();
        return ok;
    }
    return send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_SET_TENSOR, input.data(), input.size());
}

int ggml_backend_rpc_buffer_cache_query(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        size_t offset, size_t size, uint64_t hash) {
    GGML_UNUSED(size);
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *) buffer->context;
    rpc_msg_set_tensor_hash_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.hash   = hash;
    rpc_msg_set_tensor_hash_rsp response;
    if (!send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response))) {
        return -1;
    }
    return response.result ? 1 : 0;
}

bool ggml_backend_rpc_buffer_cache_upload(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    return rpc_buffer_set_tensor_raw(buffer, tensor, data, offset, size);
}

const char * ggml_backend_rpc_buffer_endpoint(ggml_backend_buffer_t buffer) {
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    ggml_backend_rpc_buffer_type_context * ctx = (ggml_backend_rpc_buffer_type_context *) buft->context;
    return ctx->endpoint.c_str();
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (size > HASH_THRESHOLD && g_rpc_client_cache.load(std::memory_order_relaxed)) {
        int result = ggml_backend_rpc_buffer_cache_query(buffer, tensor, offset, size, fnv_hash((const uint8_t *) data, size));
        RPC_STATUS_ASSERT(result >= 0);
        if (result > 0) {
            return;
        }
    }
    RPC_STATUS_ASSERT(rpc_buffer_set_tensor_raw(buffer, tensor, data, offset, size));
}

// defined with the other wire-bf16 helpers, further down next to the async lanes
static bool rpc_wire_bf16_ok(const ggml_tensor * tensor, uint64_t offset, size_t size);

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    // this is the blocking point of a synchronous stage decode: the server
    // answers only after every command queued ahead of it -- including the
    // graph -- has run, so this call's wall time is remote compute + wire.
    const bool trace = rpc_wire_trace_enabled();
    rpc_wire_ep_stat * st = trace ? rpc_wire_stat(ctx->endpoint) : nullptr;
    const int64_t t0 = trace ? ggml_time_us() : 0;
    // [fork] GGML_RPC_WIRE_BF16 used to reach only the async lanes, so the one
    // path that needs it most never saw it: SPD requires GGML_RPC_ASYNC=0, and
    // then set/get_tensor_async both fall back here. On a 9-stage split every
    // board answers its boundary read within about a millisecond of the others,
    // so ~512 KiB lands on the head's single gigabit NIC in a burst each step --
    // and board compute is steady to p99 while the step time is not. The server
    // side of GET_TENSOR_BF16 has been deployed since the wire-bf16 branch.
    const bool wire_bf16 = rpc_wire_bf16_ok(tensor, offset, size);
    bool status;
    if (wire_bf16) {
        std::vector<uint8_t> wire(size / 2);
        status = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_GET_TENSOR_BF16,
                                      &request, sizeof(request), wire.data(), wire.size());
        if (status) {
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) wire.data(),
                                  (float *) data, size / sizeof(float));
        }
    } else {
        status = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    }
    if (trace) {
        st->get_us    += (uint64_t) (ggml_time_us() - t0);
        st->get_bytes += wire_bf16 ? size / 2 : size;
        st->get_n     += 1;
        rpc_wire_trace_tick();
    }
    RPC_STATUS_ASSERT(status);
}

// [fork] client half of RPC_CMD_IMATRIX_SQSUM -- see the enum comment. Returns
// false (and collects nothing) whenever the tensor is not RPC-resident or the
// daemon predates patch 3, so the caller can fall back to the plain GET path.
// Exported block-form from the .cpp: a fork entry point must never cost a
// public-header change.
extern "C" {
GGML_BACKEND_API bool ggml_backend_rpc_imatrix_sqsum(
        const struct ggml_tensor * src1,
        const struct ggml_tensor * ids,
        int64_t                    n_mat,
        int64_t                    src0_ne2,
        int64_t                    src0_ne3,
        float                    * sums,
        int64_t                  * counts) {
    if (src1 == nullptr || src1->buffer == nullptr || !ggml_backend_buffer_is_rpc(src1->buffer)) {
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || n_mat <= 0 || sums == nullptr) {
        return false;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *) src1->buffer->context;
    if (rpc_server_patch(ctx->endpoint) < GGML_RPC_IMAT_MIN_PATCH) {
        return false;
    }
    if (ids != nullptr && (ids->buffer == nullptr || !ggml_backend_buffer_is_rpc(ids->buffer))) {
        return false; // ids must be readable by the same daemon
    }
    if (ids != nullptr) {
        ggml_backend_rpc_buffer_context * ictx = (ggml_backend_rpc_buffer_context *) ids->buffer->context;
        if (ictx->endpoint != ctx->endpoint) {
            return false;
        }
    }

    rpc_msg_imatrix_sqsum_req request = {};
    request.src1     = serialize_tensor(src1);
    request.n_mat    = (uint32_t) n_mat;
    request.has_ids  = ids != nullptr ? 1u : 0u;
    request.src0_ne2 = (uint32_t) (src0_ne2 > 0 ? src0_ne2 : 1);
    request.src0_ne3 = (uint32_t) (src0_ne3 > 0 ? src0_ne3 : 1);
    if (ids != nullptr) {
        request.ids = serialize_tensor(ids);
    }

    const int64_t ne0 = src1->ne[0];
    std::vector<uint8_t> response(sizeof(uint64_t)*n_mat + sizeof(float)*ne0*n_mat);

    const bool trace = rpc_wire_trace_enabled();
    rpc_wire_ep_stat * st = trace ? rpc_wire_stat(ctx->endpoint) : nullptr;
    const int64_t t0 = trace ? ggml_time_us() : 0;
    const bool ok = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_IMATRIX_SQSUM,
                                         &request, sizeof(request), response.data(), response.size());
    if (trace) {
        st->get_us    += (uint64_t) (ggml_time_us() - t0);
        st->get_bytes += response.size();
        st->get_n     += 1;
        rpc_wire_trace_tick();
    }
    if (!ok) {
        return false;
    }

    const uint64_t * rc = (const uint64_t *) response.data();
    const float    * rs = (const float *) (response.data() + sizeof(uint64_t)*n_mat);
    std::memcpy(sums, rs, sizeof(float)*ne0*n_mat);
    if (counts != nullptr) {
        for (int64_t m = 0; m < n_mat; ++m) {
            counts[m] = (int64_t) rc[m];
        }
    }
    return true;
}
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->endpoint != dst_ctx->endpoint) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd_ordered(ctx->endpoint, get_socket(ctx->endpoint), RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    if (sock == nullptr) {
        // report as an allocation failure so the caller errors out cleanly
        GGML_LOG_ERROR("[%s] lost connection to %s\n", __func__, buft_ctx->endpoint.c_str());
        return nullptr;
    }
    bool status = send_rpc_cmd_ordered(buft_ctx->endpoint, sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{buft_ctx->endpoint, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::string & endpoint, const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd_ordered(endpoint, sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::string & endpoint, const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd_ordered(endpoint, sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // [fork, pipeline-prefill] cache the remote responses, keyed by shapes.
        // galloc asks for every FLASH_ATTN_EXT / MUL_MAT_ID node on every graph
        // allocation, and the ordered round trip queues behind in-flight compute
        // on the endpoint's socket - under pipeline parallelism that serialized
        // prefill to one full pipeline drain per ubatch. Shapes repeat across
        // ubatches, so the steady state becomes fully local.
        static std::mutex alloc_size_cache_mutex;
        static std::unordered_map<std::string, uint64_t> alloc_size_cache;

        std::string key;
        key.reserve(512);
        key += buft_ctx->endpoint;
        {
            char dev_buf[16];
            snprintf(dev_buf, sizeof(dev_buf), "#%u", buft_ctx->device);
            key += dev_buf;
        }
        auto append_tensor = [&key](const ggml_tensor * t) {
            if (t == nullptr) {
                key += "|-";
                return;
            }
            char buf[192];
            snprintf(buf, sizeof(buf), "|%d:%d:%lld,%lld,%lld,%lld:%zu,%zu,%zu,%zu",
                     (int) t->type, (int) t->op,
                     (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
                     t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
            key += buf;
        };
        append_tensor(tensor);
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            append_tensor(tensor->src[i]);
        }

        {
            std::lock_guard<std::mutex> lock(alloc_size_cache_mutex);
            auto it = alloc_size_cache.find(key);
            if (it != alloc_size_cache.end()) {
                return it->second;
            }
        }

        auto sock = get_socket(buft_ctx->endpoint);
        if (sock == nullptr) {
            // best-effort fallback; a dead endpoint will fail the subsequent
            // alloc with a clean error anyway
            GGML_LOG_ERROR("[%s] lost connection to %s\n", __func__, buft_ctx->endpoint.c_str());
            return ggml_nbytes(tensor);
        }

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd_ordered(buft_ctx->endpoint, sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        {
            std::lock_guard<std::mutex> lock(alloc_size_cache_mutex);
            alloc_size_cache.emplace(key, response.alloc_size);
        }

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    // NOTE: do not flush the coalescing buffer here. It looks like the natural
    // place ("synchronize means everything I asked for has been sent") and it
    // is not needed -- a buffered frame is always followed by a graph submit or
    // a response-bearing command, both of which flush. It is also on the decode
    // hot path: get_socket() and the queue map are both process-global, and
    // taking them per synchronize across nine concurrent stage threads cost
    // 9.10 -> 8.1 t/s on the DSV4 SPD split (measured 2026-07-31, with the
    // buffering itself disabled, which is how it was isolated).
    if (!rpc_async_enabled()) {
        // legacy path: graph_compute is a blocking send and there are no async
        // ops in flight, so nothing to wait for
        return;
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    const std::string endpoint = rpc_ctx->endpoint;
    const uint32_t device = rpc_ctx->device;
    // full-duplex lanes: drain the lane workers first so every lane command is
    // on the wire, then let the main-lane barrier below cover their counts via
    // a LANE_FENCE + ping (the fence makes the server wait until both lanes
    // fully processed everything submitted so far, the ping proves it did).
    rpc_ep_lanes * ep = get_ep_lanes(endpoint);
    rpc_stream * lane_set_st = nullptr;
    rpc_stream * lane_get_st = nullptr;
    uint64_t lane_set_cnt = 0, lane_get_cnt = 0;
    bool lanes_dirty = false;
    {
        std::lock_guard<std::mutex> l(ep->m);
        if (ep->state == 1) {
            lane_set_st  = ep->set_stream;
            lane_get_st  = ep->get_stream;
            lane_set_cnt = ep->set_enq;
            lane_get_cnt = ep->get_enq;
            lanes_dirty  = lane_set_cnt > ep->barrier_set || lane_get_cnt > ep->barrier_get;
        }
    }
    if (lane_set_st != nullptr) {
        lane_set_st->drain();
    }
    if (lane_get_st != nullptr) {
        lane_get_st->drain();
    }
    rpc_stream * stream = get_stream(endpoint);
    // make sure the server has drained every in-order command (incl. the last
    // fire-and-forget GRAPH_COMPUTE), then wait for all worker tasks to finish.
    // when nothing was enqueued since the last completed ping the socket is
    // quiescent and the round-trip is skipped - llama_context::synchronize()
    // fires per sched x per backend and would otherwise storm the LAN with
    // redundant pings (the dominant stage-2 spec_proc overhead).
    if (stream->needs_barrier() || lanes_dirty) {
        uint64_t covered;
        {
            // fence accounting mirrors rpc_main_enqueue_counted, but the task
            // goes through enqueue_barrier so the ping does not mark the
            // stream dirty again
            std::lock_guard<std::mutex> l(ep->m);
            const bool fence = ep->state == 1 && (ep->set_enq != ep->fenced_set || ep->get_enq != ep->fenced_get);
            rpc_msg_lane_fence_req freq = { ep->set_enq, ep->get_enq };
            if (fence) {
                ep->fenced_set = ep->set_enq;
                ep->fenced_get = ep->get_enq;
                ep->main_enq++;
            }
            ep->main_enq++;
            covered = stream->enqueue_barrier([endpoint, device, fence, freq]{
                if (fence) {
                    auto sock = get_socket(endpoint);
                    if (sock != nullptr) {
                        send_rpc_cmd(sock, RPC_CMD_LANE_FENCE, &freq, sizeof(freq));
                    }
                }
                rpc_ping(endpoint, device);
            });
        }
        stream->drain();
        stream->mark_barrier(covered);
        {
            std::lock_guard<std::mutex> l(ep->m);
            if (lane_set_cnt > ep->barrier_set) {
                ep->barrier_set = lane_set_cnt;
            }
            if (lane_get_cnt > ep->barrier_get) {
                ep->barrier_get = lane_get_cnt;
            }
        }
    } else {
        stream->drain();
    }
}

static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    // iterative post-order DFS: recursing per src/view_src overflows the stack
    // on graphs whose longest dependency chain spans thousands of tensors
    // (e.g. recurrent-state models under --split-mode tensor)
    std::vector<std::pair<ggml_tensor *, bool>> stack;
    stack.push_back({tensor, false});
    while (!stack.empty()) {
        auto [t, expanded] = stack.back();
        stack.pop_back();
        if (expanded) {
            rpc_tensor result = serialize_tensor(t);
            const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, t);
            if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
                result.use_count = cgraph->use_counts[hash_pos];
            }
            tensors.push_back(result);
            continue;
        }
        if (t == nullptr || visited.find(t) != visited.end()) {
            continue;
        }
        visited.insert(t);
        // pop order must mirror the recursive version: src[0..n], view_src, then t itself
        stack.push_back({t, true});
        stack.push_back({t->view_src, false});
        for (int i = GGML_MAX_SRC - 1; i >= 0; i--) {
            stack.push_back({t->src[i], false});
        }
    }
}

// Diagnostic: flag any leaf in a subgraph bound for `endpoint` whose data lives
// in an RPC buffer owned by a DIFFERENT endpoint. The server rejects such a
// tensor with "[create_node] invalid data ptr" (buffer==null && data!=null)
// because RPC is star-only: there is no remote<->remote path, so a board can
// never dereference another board's buffer. Gated by LLAMA_RPC_TRACE_XDEV.
static const char * rpc_xdev_ep(const ggml_tensor * t) {
    if (t == nullptr || t->buffer == nullptr || !ggml_backend_buffer_is_rpc(t->buffer)) {
        return nullptr;
    }
    auto * bctx = (ggml_backend_rpc_buffer_context *) t->buffer->context;
    return bctx ? bctx->endpoint.c_str() : nullptr;
}

static void rpc_trace_xdev(const std::string & endpoint, const ggml_cgraph * cgraph) {
    static const bool on = getenv("LLAMA_RPC_TRACE_XDEV") != nullptr;
    if (!on) {
        return;
    }
    // The nodes in cgraph->nodes[] are exactly what serialize_graph ships to the
    // server. A node (or any of its srcs) whose buffer belongs to a different RPC
    // endpoint is what the server rejects as "invalid data ptr". Report each with
    // its view_src root so we can see whether it is a foreign VIEW, a foreign real
    // src, or an orphan foreign node with no in-graph consumer.
    std::unordered_set<const ggml_tensor *> consumed; // referenced as some node's src
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (n->src[j]) consumed.insert(n->src[j]);
        }
    }
    auto root_of = [](const ggml_tensor * t) {
        while (t->view_src) t = t->view_src;
        return t;
    };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * n = cgraph->nodes[i];
        const char * nep = rpc_xdev_ep(n);
        if (nep && endpoint != nep) {
            const ggml_tensor * r = root_of(n);
            GGML_LOG_ERROR("[rpc xdev] target=%s node[%d] '%s' (op=%s) FOREIGN@%s%s"
                           " view_root='%s'@%s ne=[%lld,%lld]\n",
                endpoint.c_str(), i, n->name, ggml_op_name(n->op), nep,
                consumed.count(n) ? "" : " ORPHAN(no-consumer)",
                r->name, rpc_xdev_ep(r) ? rpc_xdev_ep(r) : "local",
                (long long)n->ne[0], (long long)n->ne[1]);
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const ggml_tensor * s = n->src[j];
            const char * sep = rpc_xdev_ep(s);
            if (sep && endpoint != sep) {
                const ggml_tensor * r = root_of(s);
                GGML_LOG_ERROR("[rpc xdev] target=%s node[%d] '%s' (op=%s) has FOREIGN src[%d] '%s'"
                               " (op=%s)@%s view_root='%s'@%s\n",
                    endpoint.c_str(), i, n->name, ggml_op_name(n->op), j, s->name,
                    ggml_op_name(s->op), sep, r->name, rpc_xdev_ep(r) ? rpc_xdev_ep(r) : "local");
            }
        }
    }
}

// True when the tensor's data lives in an RPC buffer owned by a DIFFERENT
// endpoint than the one we are serializing this subgraph for.
static bool rpc_node_is_foreign(const ggml_tensor * t, const std::string & endpoint) {
    const char * ep = rpc_xdev_ep(t);
    return ep != nullptr && endpoint != ep;
}

// [fork] Collect and emit are split so the caller can gather the node/tensor set
// once, derive a content uid from it, and only pay for the payload on a miss.
static void collect_graph(const std::string & endpoint, const ggml_cgraph * cgraph,
                          std::vector<ggml_tensor *> & nodes, std::vector<rpc_tensor> & tensors) {
    // Drop nodes whose data lives on a different RPC endpoint before shipping the
    // subgraph. RPC is star-only — there is no remote<->remote memory path — so a
    // board can never dereference another board's buffer; serializing such a node
    // makes the server reject it ("[create_node] invalid data ptr": data!=0,
    // buffer==null) and abort. These nodes are always dead here: the backend
    // scheduler (ggml_backend_sched pass 5) already rewired every LIVE consumer to
    // a host-routed cross-backend copy, leaving only orphaned view nodes — e.g.
    // DeepSeek-V4's per-stream HC residual slices (build_hc_pre/post view the
    // previous board's l_out) that fall inside this board's contiguous split range.
    // Skipping them is loss-free (nothing kept references them) and keeps the graph
    // uid -> filtered-node-set mapping deterministic, so the server's graph cache
    // stays consistent across recompute.
    nodes.clear();
    tensors.clear();
    nodes.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (rpc_node_is_foreign(cgraph->nodes[i], endpoint)) {
            continue;
        }
        nodes.push_back(cgraph->nodes[i]);
    }
    std::unordered_set<ggml_tensor*> visited;
    for (size_t i = 0; i < nodes.size(); i++) {
        add_tensor(nodes[i], cgraph, tensors, visited);
    }
}

// [fork] GGML_RPC_STABLE_UID=1: derive the graph uid from what the server
// actually stores -- the filtered node list and the tensor descriptors -- rather
// than from ggml's monotonic counter, which hands every rebuild a fresh id and so
// can never hit the server's graph cache.
//
// This is what makes pipelined decode affordable. Pipelining requires graph reuse
// to be OFF (the reuse path drains the pipeline before set_inputs), and with it
// off every ubatch was re-shipping the full node list + ~1300 tensor descriptors
// to every stage. A rebuilt-but-identical graph now costs a 16-byte RECOMPUTE.
//
// Safety rests entirely on the uid covering every byte serialize_collected emits:
// any change of shape, op param, buffer offset, data pointer or node set yields a
// different uid and falls back to a full send. Per-token differences (token ids,
// positions, masks, the DSV4 compressed-state index inputs) travel as tensor DATA
// via SET_TENSOR, not as graph structure, which is exactly why the same stored
// graph can be recomputed for them -- the same contract the existing reuse path
// already relies on. rpc_tensor is fully initialized (serialize_tensor memsets
// name and padding), so hashing it raw is deterministic.
static uint64_t graph_content_uid(const std::vector<ggml_tensor *> & nodes,
                                  const std::vector<rpc_tensor> & tensors) {
    auto mix = [](uint64_t h, const void * p, size_t bytes) {
        const uint64_t * w = (const uint64_t *) p;
        for (size_t i = 0; i < bytes/sizeof(uint64_t); i++) {
            h ^= w[i] + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        }
        return h;
    };

    // The `id`/`src`/`view_src` fields are HOST POINTERS to the ggml_tensor structs
    // of this build. Rebuilding an identical graph allocates fresh structs at fresh
    // addresses, so hashing them raw makes every graph unique and the cache can
    // never hit. They also do not affect execution: the server uses them only to
    // rewire the DAG at deserialize time, and the tensors it stores carry their own
    // copies. So references are canonicalised to indices into `tensors` (add_tensor
    // walks a fixed-order DFS, so the order is deterministic for a given topology)
    // while everything that DOES affect execution -- type, buffer, ne/nb, op,
    // op_params, flags, view_offs and the device `data` pointer -- is hashed
    // verbatim. A graph that would compute anything different therefore cannot
    // collide with one already on the server.
    std::unordered_map<uint64_t, uint64_t> index_of;
    index_of.reserve(tensors.size()*2);
    for (size_t i = 0; i < tensors.size(); i++) {
        index_of[tensors[i].id] = i;
    }
    const auto ref = [&](uint64_t id) -> uint64_t {
        if (id == 0) {
            return UINT64_MAX;
        }
        auto it = index_of.find(id);
        return it == index_of.end() ? UINT64_MAX - 1 : it->second;
    };

    uint64_t h = 1469598103934665603ull;
    const uint64_t counts[2] = { (uint64_t) nodes.size(), (uint64_t) tensors.size() };
    h = mix(h, counts, sizeof(counts));

    for (const ggml_tensor * n : nodes) {
        const uint64_t idx = ref((uint64_t) (uintptr_t) n);
        h = mix(h, &idx, sizeof(idx));
    }

    for (const rpc_tensor & t : tensors) {
        uint64_t refs[GGML_MAX_SRC + 1];
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            refs[i] = ref(t.src[i]);
        }
        refs[GGML_MAX_SRC] = ref(t.view_src);
        h = mix(h, refs, sizeof(refs));

        // every execution-affecting field, pointer identity excluded
        const uint64_t scalars[4] = { t.buffer, t.view_offs, t.data,
                                      ((uint64_t) t.type << 32) | (uint64_t) t.op };
        h = mix(h, scalars, sizeof(scalars));
        h = mix(h, t.ne, sizeof(t.ne));
        h = mix(h, t.nb, sizeof(t.nb));
        h = mix(h, t.op_params, sizeof(t.op_params));
        const uint64_t flags = (uint64_t) (uint32_t) t.flags;
        h = mix(h, &flags, sizeof(flags));
    }

    return h == 0 ? 1 : h; // uid 0 means "do not cache" in the protocol
}

// [fork] The content uid above is exact but costs a DFS + ~1300 descriptor
// serializations per split -- ~1 ms, ten splits per token on the nine-board
// House, so a pipelined decode paid ~10 ms of client CPU per level just to
// discover that the graph was the one it shipped last time. This fingerprint
// hashes the raw ggml_tensor structs of everything serialize_collected reads
// (each node, its srcs, and their view chains, plus the RPC buffer's remote id
// that serialize_tensor dereferences) without allocating. Equal fingerprints
// mean byte-identical serialized graphs, so the uid can be looked up instead
// of recomputed; a fingerprint never seen (or evicted) falls back to the exact
// path, so a false miss only costs time, never correctness.
static inline uint64_t rpc_fp_mix(uint64_t h, const void * p, size_t bytes) {
    const uint64_t * w = (const uint64_t *) p;
    for (size_t i = 0; i < bytes/sizeof(uint64_t); i++) {
        h = (h ^ w[i]) * 0x9E3779B97F4A7C15ull;
        h ^= h >> 29;
    }
    return h;
}

// only the fields serialize_tensor reads: the name and the CPU-side fields
// (extra, padding) never reach the wire, and at ~730 nodes x 3 structs per
// split every byte hashed is paid ten times per token
static inline uint64_t rpc_fp_tensor(uint64_t h, const ggml_tensor * t) {
    const uint64_t ptr = (uint64_t) (uintptr_t) t;
    h = rpc_fp_mix(h, &ptr, sizeof(ptr));
    if (t == nullptr) {
        return h;
    }
    uint64_t remote = 0;
    if (t->buffer && ggml_backend_buffer_is_rpc(t->buffer)) {
        auto * ctx = (ggml_backend_rpc_buffer_context *) t->buffer->context;
        remote = ctx ? ctx->remote_ptr : 0;
    }
    const uint64_t head[4] = {
        ((uint64_t) (uint32_t) t->type << 32) | (uint64_t) (uint32_t) t->op,
        (uint64_t) (uintptr_t) t->buffer, remote, (uint64_t) (uintptr_t) t->data,
    };
    h = rpc_fp_mix(h, head, sizeof(head));
    h = rpc_fp_mix(h, t->ne, sizeof(t->ne));
    h = rpc_fp_mix(h, t->nb, sizeof(t->nb));
    h = rpc_fp_mix(h, t->op_params, sizeof(t->op_params));
    h = rpc_fp_mix(h, t->src, sizeof(t->src));
    const uint64_t tail[3] = { (uint64_t) (uint32_t) t->flags, (uint64_t) (uintptr_t) t->view_src, (uint64_t) t->view_offs };
    return rpc_fp_mix(h, tail, sizeof(tail));
}

// What a stored graph computes on recompute is fixed by its nodes and their
// direct srcs (op, params, shapes, strides, device data pointers): deeper
// tensors reach the wire only as leaf descriptors the server never executes.
// So the fingerprint covers exactly node + srcs. Two graphs that serialize
// differently further up (a view chain, an input's own producers) but agree
// here compute the same thing, and hashing 13 structs per node instead of
// ~3 cost 0.6 ms per split - as much as the walk it was meant to replace.
static uint64_t rpc_graph_quick_fp(const ggml_cgraph * cgraph) {
    uint64_t h = 0x243F6A8885A308D3ull;
    const uint64_t n = cgraph->n_nodes;
    h = rpc_fp_mix(h, &n, sizeof(n));
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        h = rpc_fp_tensor(h, node);
        for (int k = 0; k < GGML_MAX_SRC; k++) {
            if (node->src[k] != nullptr) {
                h = rpc_fp_tensor(h, node->src[k]);
            }
        }
    }
    return h == 0 ? 1 : h;
}

static void serialize_collected(uint32_t device, uint64_t uid,
                                const std::vector<ggml_tensor *> & nodes,
                                const std::vector<rpc_tensor> & tensors,
                                std::vector<uint8_t> & output) {
    uint32_t n_nodes = nodes.size();
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &uid, sizeof(uid));
    dest += sizeof(uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static void serialize_graph(uint32_t device, uint64_t uid, const std::string & endpoint,
                            const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    std::vector<ggml_tensor *> nodes;
    std::vector<rpc_tensor>    tensors;
    collect_graph(endpoint, cgraph, nodes, tensors);
    serialize_collected(device, uid, nodes, tensors, output);
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    const std::string endpoint = rpc_ctx->endpoint;
    const uint32_t device = rpc_ctx->device;
    const bool async = rpc_async_enabled();

    const bool wire_trace = rpc_wire_trace_enabled();
    rpc_wire_ep_stat * wire_st = wire_trace ? rpc_wire_stat(endpoint) : nullptr;
    const int64_t wire_t0 = wire_trace ? ggml_time_us() : 0;
    if (wire_trace) {
        // a pipelined decode may go a long time with no traced set/get (its
        // inputs and reads take the async paths), so flush from here too
        rpc_wire_trace_tick();
    }
    struct wire_submit_scope {
        rpc_wire_ep_stat * st; int64_t t0;
        ~wire_submit_scope() {
            if (st != nullptr) {
                st->gc_us += (uint64_t) (ggml_time_us() - t0);
                st->gc_n  += 1;
            }
        }
    } wire_scope { wire_st, wire_t0 };

    // [fork] see graph_content_uid(): a content-derived uid lets a rebuilt but
    // structurally identical graph hit the server cache, which is what a
    // reuse-disabled (i.e. pipelined) decode needs to stay affordable.
    static const bool stable_uid = [] {
        const char * e = getenv("GGML_RPC_STABLE_UID");
        return e && atoi(e) != 0;
    }();

    std::vector<ggml_tensor *> nodes;
    std::vector<rpc_tensor>    tensors;
    uint64_t uid = cgraph->uid;
    bool collected = false;
    if (stable_uid) {
        // [fork] A split's ggml_cgraph::uid is assigned fresh by every
        // ggml_backend_sched_split_graph, and every mutation of the split's
        // node/tensor structs (src rewiring, gallocr data pointers) happens in
        // that same ggml_backend_sched_alloc_graph before the first compute.
        // A graph the scheduler re-submits unchanged (llama's reuse path:
        // set_inputs writes tensor DATA only) therefore carries the same
        // split uid and the same content uid, so the content uid can be
        // looked up by split uid instead of re-fingerprinting ~730 nodes x 4
        // structs per split -- which on a ten-split pipelined decode level was
        // most of the submit cost (~0.3 ms per split on the Xeon 6138 head).
        // The fingerprint below stays as the fallback for a split instance
        // seen for the first time, so a rebuilt-but-identical graph still
        // resolves to the uid the server already holds.
        static const bool split_off = getenv("GGML_RPC_SPLIT_UID_OFF") != nullptr;
        static const bool quick_off = getenv("GGML_RPC_QUICK_UID_OFF") != nullptr;
        const int64_t t_key0 = wire_trace ? ggml_time_us() : 0;
        auto & split_map = rpc_dev_ctx->split_uids;
        const uint64_t split_key = split_off ? 0 : cgraph->uid;
        auto sit = split_key != 0 ? split_map.find(split_key) : split_map.end();
        if (sit != split_map.end()) {
            uid = sit->second;
            if (wire_st != nullptr) {
                wire_st->gc_split_hit += 1;
            }
        } else {
            auto & quick = rpc_dev_ctx->quick_uids;
            const uint64_t fp = quick_off ? 0 : rpc_graph_quick_fp(cgraph);
            auto it = fp != 0 ? quick.find(fp) : quick.end();
            if (it != quick.end()) {
                uid = it->second;
            } else {
                collect_graph(endpoint, cgraph, nodes, tensors);
                uid = graph_content_uid(nodes, tensors);
                collected = true;
                if (fp != 0) {
                    if (quick.size() >= 4096) {
                        quick.clear();
                    }
                    quick[fp] = uid;
                }
            }
            if (split_key != 0) {
                if (split_map.size() >= 4096) {
                    split_map.clear();
                }
                split_map[split_key] = uid;
            }
        }
        if (wire_st != nullptr) {
            wire_st->gc_key_us += (uint64_t) (ggml_time_us() - t_key0);
        }
    }

    // LRU over the uids the server holds deserialized for this device. A hit
    // means the server can recompute without a serialize/deserialize round.
    //
    // [fork] GGML_RPC_GRAPH_SLOTS: the default 32 was sized for one graph shape
    // per ubatch. Under a split decode the working set is (sequences in flight) x
    // (GGML_SCHED_COPIES) distinct graphs -- each sequence views its own KV stream
    // and each rotates through the pipeline copy slots -- so 16 slots and 6 copies
    // want ~96 and thrash a 32-entry cache down to a ~1% hit rate. Each cached
    // graph costs the daemon a deserialized node/tensor set (~0.5 MB here).
    static const size_t graph_slots = [] {
        const char * e = getenv("GGML_RPC_GRAPH_SLOTS");
        const int v = e ? atoi(e) : 0;
        return v > 0 ? (size_t) v : (size_t) RPC_GRAPH_CACHE_SLOTS;
    }();

    bool reuse = false;
    uint64_t evicted_uid = 0;
    if (uid != 0) {
        auto & known = rpc_dev_ctx->known_graph_uids;
        auto it = std::find(known.begin(), known.end(), uid);
        if (it != known.end()) {
            known.erase(it);
            known.insert(known.begin(), uid);
            reuse = true;
        } else {
            known.insert(known.begin(), uid);
            if (known.size() > graph_slots) {
                evicted_uid = known.back();
                known.pop_back();
            }
        }
    }
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = device;
        request.uid    = uid;
        if (async) {
            // enqueue on the endpoint's stream so the scheduler thread does not block
            rpc_main_enqueue_counted(endpoint, get_stream(endpoint), 1, [endpoint, request]{
                auto sock = get_socket(endpoint);
                if (sock == nullptr) {
                    GGML_LOG_ERROR("[rpc graph_recompute] lost connection to %s\n", endpoint.c_str());
                    return;
                }
                send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
            });
            return GGML_STATUS_SUCCESS;
        }
        auto sock = get_socket(endpoint);
        if (sock == nullptr) {
            GGML_LOG_ERROR("[%s] lost connection to %s\n", __func__, endpoint.c_str());
            return GGML_STATUS_FAILED;
        }
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    } else {
        if (wire_st != nullptr) {
            wire_st->gc_full_n += 1;
        }
        rpc_trace_xdev(endpoint, cgraph);
        std::vector<uint8_t> input;
        if (stable_uid) {
            if (!collected) {
                // fingerprint hit on a uid the server has since evicted
                collect_graph(endpoint, cgraph, nodes, tensors);
                GGML_ASSERT(graph_content_uid(nodes, tensors) == uid);
            }
            serialize_collected(device, uid, nodes, tensors, input);
        } else {
            serialize_graph(device, uid, endpoint, cgraph, input);
        }
        rpc_msg_graph_forget_req forget = { device, evicted_uid };
        if (async) {
            auto in = std::make_shared<std::vector<uint8_t>>(std::move(input));
            const uint32_t n_cmds = forget.uid != 0 ? 2 : 1;
            rpc_main_enqueue_counted(endpoint, get_stream(endpoint), n_cmds, [endpoint, in, forget]{
                auto sock = get_socket(endpoint);
                if (sock == nullptr) {
                    GGML_LOG_ERROR("[rpc graph_compute] lost connection to %s\n", endpoint.c_str());
                    return;
                }
                if (forget.uid != 0) {
                    send_rpc_cmd(sock, RPC_CMD_GRAPH_FORGET, &forget, sizeof(forget));
                }
                send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, in->data(), in->size());
            });
            return GGML_STATUS_SUCCESS;
        }
        auto sock = get_socket(endpoint);
        if (sock == nullptr) {
            GGML_LOG_ERROR("[%s] lost connection to %s\n", __func__, endpoint.c_str());
            return GGML_STATUS_FAILED;
        }
        if (forget.uid != 0) {
            bool fstatus = send_rpc_cmd(sock, RPC_CMD_GRAPH_FORGET, &forget, sizeof(forget));
            RPC_STATUS_ASSERT(fstatus);
        }
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

// --- async backend ops [fork, PipeDec Phase 1] ---
//
// Tasks must capture everything they need BY VALUE at enqueue time: the graph
// tensor structs they reference live in scheduler/graph memory that may be
// rewritten for the next ubatch before the task runs on the stream worker.

// serialized SET_TENSOR wire message: | rpc_tensor | offset (8) | payload (size) |
// built on the enqueuing thread; the payload may be filled in later by a task
static const size_t RPC_SET_TENSOR_HDR = sizeof(rpc_tensor) + sizeof(uint64_t);

static std::shared_ptr<std::vector<uint8_t>> rpc_prepare_set_tensor(
        const ggml_tensor * tensor, const void * data, uint64_t offset, size_t size) {
    rpc_tensor rt = serialize_tensor(tensor);
    auto msg = std::make_shared<std::vector<uint8_t>>(RPC_SET_TENSOR_HDR + size);
    memcpy(msg->data(), &rt, sizeof(rt));
    memcpy(msg->data() + sizeof(rt), &offset, sizeof(offset));
    if (data != nullptr) {
        memcpy(msg->data() + RPC_SET_TENSOR_HDR, data, size);
    }
    return msg;
}

// bf16 wire compression (GGML_RPC_WIRE_BF16=1): halves the wire bytes of
// ASYNC f32 tensor traffic - pipeline stage-boundary activations, INPUT
// staging (KQ masks), and deferred tap/embedding reads. Load-time weight
// uploads go through the synchronous buffer interface and are never
// compressed. bf16 keeps the f32 exponent (8 mantissa bits), so large
// activations can never overflow the way IEEE f16 would.
static bool rpc_wire_bf16_enabled() {
    static const bool on = []() {
        const char * e = std::getenv("GGML_RPC_WIRE_BF16");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    return on;
}

static bool rpc_wire_bf16_ok(const ggml_tensor * tensor, uint64_t offset, size_t size) {
    return rpc_wire_bf16_enabled()
        && tensor->type == GGML_TYPE_F32
        && offset % sizeof(float) == 0
        && size   % sizeof(float) == 0
        && size >= 4096; // tiny messages are latency-bound, not bandwidth-bound
}

static std::shared_ptr<std::vector<uint8_t>> rpc_prepare_set_tensor_bf16(
        const ggml_tensor * tensor, const void * data, uint64_t offset, size_t size) {
    rpc_tensor rt = serialize_tensor(tensor);
    auto msg = std::make_shared<std::vector<uint8_t>>(RPC_SET_TENSOR_HDR + size / 2);
    memcpy(msg->data(), &rt, sizeof(rt));
    memcpy(msg->data() + sizeof(rt), &offset, sizeof(offset));
    if (data != nullptr) {
        ggml_fp32_to_bf16_row((const float *) data,
                              (ggml_bf16_t *) (msg->data() + RPC_SET_TENSOR_HDR),
                              size / sizeof(float));
    }
    return msg;
}

// ---------------------------------------------------------------------------
// Synchronous peer push  [fork, SPD]
//
// The SPD stage pipeline must run with GGML_RPC_ASYNC=0 (async loses 2.65x on
// its decode and 1.7x on its prefill), which keeps rpc_lanes_get_active() null
// and with it the scheduler's peer-transfer path. But its stage boundaries are
// 16.8 MB f32 per 256-token chunk on DSV4, hairpinned through the client's
// per-board ~1 Gbit links serialized against compute -- measured ~290 ms of
// every ~1.23 s prefill cell.
//
// These entry points drive the DEPLOYED server peer machinery from a sync
// client -- SESSION_INFO / PEER_OPEN / LANE_ATTACH(GET) / PUSH_TENSOR /
// LANE_FENCE, no daemon change -- with ordering enforced by the caller's
// command sequence instead of the async lane counters:
//   - a push is only issued after the producer's graph is known complete
//     (any response-bearing command on its main socket after GRAPH_COMPUTE
//     proves that; SPD's anchor readback inside llama_decode qualifies);
//   - a push lands in a tensor nothing reads concurrently (caller-owned
//     staging, consumed via an on-server COPY_TENSOR);
//   - LANE_FENCE{set = push ordinal} on the consumer's main socket stalls its
//     command loop until the peer applier has applied that many pushes, so
//     the following COPY_TENSOR reads the delivered payload.
// Threading contract: prepare/fence run on the thread that owns the consumer
// endpoint's main socket, push on the thread that owns the producer's. The
// push lane is a dedicated GET-lane connection used for nothing else.
// ---------------------------------------------------------------------------

struct rpc_sync_peer_state {
    std::mutex m;
    std::unordered_map<std::string, uint64_t>   session_ids;  // endpoint -> session id
    std::unordered_map<std::string, socket_ptr> push_lanes;   // producer endpoint -> GET lane
    std::unordered_map<std::string, uint64_t>   push_counts;  // consumer endpoint -> pushes issued
    // deferred-ack bookkeeping per PRODUCER endpoint: lane commands sent vs
    // acks drained. The ack is the producer's local read completing -- costly
    // to block on (a device read serialized into the cell), so it is drained
    // lazily; ggml_backend_rpc_sync_peer_guard orders the producer's next
    // graph behind the read with a get-lane fence instead.
    std::unordered_map<std::string, uint64_t>   lane_sent;
    std::unordered_map<std::string, uint64_t>   lane_acked;
};

static rpc_sync_peer_state & rpc_sync_peer() {
    static rpc_sync_peer_state st;
    return st;
}

static const char * rpc_tensor_endpoint(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensor->buffer)) {
        return nullptr;
    }
    auto * ctx = (ggml_backend_rpc_buffer_context *) tensor->buffer->context;
    return ctx->endpoint.c_str();
}

// SESSION_INFO over the endpoint's main socket; memoized. Must be called from
// the thread that owns that socket.
static bool rpc_sync_session_for(const std::string & endpoint, uint64_t & out) {
    auto & st = rpc_sync_peer();
    {
        std::lock_guard<std::mutex> l(st.m);
        auto it = st.session_ids.find(endpoint);
        if (it != st.session_ids.end()) {
            out = it->second;
            return true;
        }
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        return false;
    }
    rpc_msg_session_info_rsp info = {};
    if (!send_rpc_cmd(sock, RPC_CMD_SESSION_INFO, nullptr, 0, &info, sizeof(info))) {
        return false;
    }
    std::lock_guard<std::mutex> l(st.m);
    st.session_ids[endpoint] = info.session_id;
    out = info.session_id;
    return true;
}

// PEER_OPEN from the producer's main socket, plain sync send (the async
// rpc_peer_route_ready goes through the endpoint stream, which in sync mode
// would race the caller thread's own traffic on that socket). Shares the
// per-pair memo with the async path.
static bool rpc_sync_peer_route_ready(const std::string & src_endpoint,
                                      const std::string & dst_endpoint,
                                      uint64_t dst_session_id) {
    const std::string key = src_endpoint + ">" + dst_endpoint;
    {
        std::lock_guard<std::mutex> l(g_peer_route_m);
        auto it = g_peer_routes.find(key);
        if (it != g_peer_routes.end()) {
            return it->second;
        }
    }
    const std::string & via = rpc_peer_addr(dst_endpoint);
    bool ok = false;
    if (via.size() >= RPC_ENDPOINT_MAX) {
        GGML_LOG_WARN("[rpc peer] endpoint '%s' too long to route\n", via.c_str());
    } else if (rpc_server_patch(src_endpoint) < GGML_RPC_PEER_MIN_PATCH ||
               rpc_server_patch(dst_endpoint) < GGML_RPC_PEER_MIN_PATCH) {
        GGML_LOG_WARN("[rpc peer] %s -> %s: server too old for peer transfer\n",
                      src_endpoint.c_str(), dst_endpoint.c_str());
    } else {
        rpc_msg_peer_open_req req = {};
        req.session_id = dst_session_id;
        memcpy(req.endpoint, via.c_str(), via.size());
        rpc_msg_peer_open_rsp rsp = { 0 };
        auto sock = get_socket(src_endpoint);
        if (sock != nullptr) {
            ok = send_rpc_cmd(sock, RPC_CMD_PEER_OPEN, &req, sizeof(req), &rsp, sizeof(rsp)) && rsp.ok != 0;
        }
        if (ok) {
            GGML_LOG_INFO("[rpc peer] %s -> %s: direct transfer (sync client)\n",
                          src_endpoint.c_str(), dst_endpoint.c_str());
        } else {
            GGML_LOG_WARN("[rpc peer] %s -> %s: peer lane refused, using the client hairpin\n",
                          src_endpoint.c_str(), dst_endpoint.c_str());
        }
    }
    std::lock_guard<std::mutex> l(g_peer_route_m);
    g_peer_routes[key] = ok;
    return ok;
}

#if defined(_WIN32)
#    define GGML_RPC_SYNC_PEER_API extern "C" __declspec(dllexport)
#else
#    define GGML_RPC_SYNC_PEER_API extern "C" __attribute__((visibility("default")))
#endif

// [fork, chained decode] read fences: ordinal-scoped completion of async
// GET_TENSOR reads on one endpoint. ggml_backend_rpc_synchronize is endpoint-
// global (it fences every submitted command, including later cohorts' graph
// computes), but a chained caller only needs its OWN reads back - and the
// server FIFO orders a read after the graph that produced its tensor, so a
// completed read also proves that graph retired. Snapshot the ordinal right
// after issuing the reads, wait on it later; work submitted after the
// snapshot is never waited on.
GGML_RPC_SYNC_PEER_API uint64_t ggml_backend_rpc_read_ordinal(ggml_backend_t backend) {
    if (!rpc_async_enabled()) {
        return 0; // reads were synchronous; nothing to wait for
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    rpc_ep_lanes * ep = rpc_lanes_get_active(rpc_ctx->endpoint);
    if (ep != nullptr) {
        return ep->get_stream->submitted_seq();
    }
    return get_stream(rpc_ctx->endpoint)->submitted_seq();
}

GGML_RPC_SYNC_PEER_API void ggml_backend_rpc_read_wait(ggml_backend_t backend, uint64_t ordinal) {
    if (!rpc_async_enabled() || ordinal == 0) {
        return;
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backend->context;
    rpc_ep_lanes * ep = rpc_lanes_get_active(rpc_ctx->endpoint);
    if (ep != nullptr) {
        ep->get_stream->wait_completed(ordinal);
        return;
    }
    get_stream(rpc_ctx->endpoint)->wait_completed(ordinal);
}

// Learn the consumer endpoint's session id. Call from the thread that owns
// that endpoint's main socket (in SPD: the consuming stage's worker).
GGML_RPC_SYNC_PEER_API bool ggml_backend_rpc_sync_peer_prepare(const struct ggml_tensor * dst_probe) {
    const char * ep = rpc_tensor_endpoint(dst_probe);
    if (ep == nullptr || !rpc_peer_enabled() || rpc_async_enabled()) {
        return false;
    }
    uint64_t session = 0;
    return rpc_sync_session_for(ep, session);
}

// Ship `src` (whole tensor) from its server straight into `dst` on another
// server. Call from the thread that owns the producer endpoint's main socket,
// only after the producer's graph is known complete, and only into a `dst`
// nothing can be reading. On success *ordinal_out is the fence target that
// guarantees delivery. Returns false when the pair cannot route (caller keeps
// the hairpin) or on a transfer error.
GGML_RPC_SYNC_PEER_API bool ggml_backend_rpc_sync_peer_push(
        const struct ggml_tensor * src, const struct ggml_tensor * dst, uint64_t * ordinal_out) {
    const char * src_ep_c = rpc_tensor_endpoint(src);
    const char * dst_ep_c = rpc_tensor_endpoint(dst);
    if (src_ep_c == nullptr || dst_ep_c == nullptr || !rpc_peer_enabled() || rpc_async_enabled()) {
        return false;
    }
    const std::string src_ep(src_ep_c);
    const std::string dst_ep(dst_ep_c);
    if (src_ep == dst_ep) {
        return false;
    }
    const size_t size = ggml_nbytes(src);
    if (size == 0 || size != ggml_nbytes(dst)) {
        return false;
    }

    auto & st = rpc_sync_peer();
    uint64_t dst_session = 0;
    {
        std::lock_guard<std::mutex> l(st.m);
        auto it = st.session_ids.find(dst_ep);
        if (it == st.session_ids.end()) {
            return false;   // consumer never called prepare
        }
        dst_session = it->second;
    }
    if (!rpc_sync_peer_route_ready(src_ep, dst_ep, dst_session)) {
        return false;
    }

    // dedicated push lane to the producer (a GET-lane attach; single-writer by
    // the threading contract, the mutex only guards the map)
    socket_ptr lane;
    {
        std::lock_guard<std::mutex> l(st.m);
        auto it = st.push_lanes.find(src_ep);
        if (it != st.push_lanes.end()) {
            lane = it->second;
        }
    }
    if (lane == nullptr) {
        uint64_t src_session = 0;
        if (!rpc_sync_session_for(src_ep, src_session)) {
            return false;
        }
        lane = rpc_lane_connect(src_ep, src_session, RPC_LANE_GET);
        if (lane == nullptr) {
            GGML_LOG_WARN("[rpc peer] %s: push lane attach failed\n", src_ep.c_str());
            return false;
        }
        std::lock_guard<std::mutex> l(st.m);
        st.push_lanes[src_ep] = lane;
    }

    const std::string & via = rpc_peer_addr(dst_ep);
    rpc_msg_push_tensor_req req = {};
    memcpy(req.endpoint, via.c_str(), std::min(via.size(), (size_t) RPC_ENDPOINT_MAX - 1));
    req.src        = serialize_tensor(src);
    req.dst        = serialize_tensor(dst);
    req.src_offset = 0;
    req.dst_offset = 0;
    req.size       = size;
    req.wait_main  = 0;   // apply on arrival: dst is caller-owned staging
    req.wait_get   = 0;
    req.bf16       = rpc_wire_bf16_ok(src, 0, size) ? 1 : 0;

    // drain any acks still owed on this lane (their local reads finished long
    // ago); a failed earlier push surfaces here and is fatal for the pair
    uint64_t pending;
    {
        std::lock_guard<std::mutex> l(st.m);
        pending = st.lane_sent[src_ep] - st.lane_acked[src_ep];
    }
    while (pending > 0) {
        rpc_msg_push_tensor_rsp late = { 0 };
        if (!recv_msg(lane, &late, sizeof(late)) || late.ok == 0) {
            GGML_LOG_ERROR("[rpc peer] deferred push ack from %s failed\n", src_ep.c_str());
            return false;
        }
        std::lock_guard<std::mutex> l(st.m);
        ++st.lane_acked[src_ep];
        --pending;
    }

    if (!send_lane_cmd(lane, RPC_CMD_PUSH_TENSOR, 0, 0, &req, sizeof(req))) {
        GGML_LOG_WARN("[rpc peer] sync push %s -> %s failed\n", src_ep.c_str(), dst_ep.c_str());
        return false;
    }
    uint64_t sent;
    {
        std::lock_guard<std::mutex> l(st.m);
        sent = ++st.lane_sent[src_ep];
    }
    if (sent == 1) {
        // block for the very first ack on the lane: it validates the whole
        // route while the caller can still fall back to the hairpin cheaply
        rpc_msg_push_tensor_rsp rsp = { 0 };
        if (!recv_msg(lane, &rsp, sizeof(rsp)) || rsp.ok == 0) {
            GGML_LOG_WARN("[rpc peer] sync push %s -> %s failed\n", src_ep.c_str(), dst_ep.c_str());
            return false;
        }
        std::lock_guard<std::mutex> l(st.m);
        ++st.lane_acked[src_ep];
    }
    uint64_t ordinal;
    {
        std::lock_guard<std::mutex> l(st.m);
        ordinal = ++st.push_counts[dst_ep];
    }
    if (ordinal_out != nullptr) {
        *ordinal_out = ordinal;
    }
    return true;
}

// Order the producer's next main-socket command (its next graph) behind every
// push it has issued: get_done on its session counts executed get-lane
// commands, which here are exactly the pushes, and each bumps only after its
// local read of the source tensor completed. Fire-and-forget. Call from the
// thread that owns the producer's main socket, before submitting work that
// could overwrite a pushed source tensor.
GGML_RPC_SYNC_PEER_API bool ggml_backend_rpc_sync_peer_guard(const struct ggml_tensor * src_probe) {
    const char * ep = rpc_tensor_endpoint(src_probe);
    if (ep == nullptr) {
        return false;
    }
    uint64_t sent;
    {
        auto & st = rpc_sync_peer();
        std::lock_guard<std::mutex> l(st.m);
        auto it = st.lane_sent.find(ep);
        sent = it == st.lane_sent.end() ? 0 : it->second;
    }
    if (sent == 0) {
        return true;
    }
    auto sock = get_socket(ep);
    if (sock == nullptr) {
        return false;
    }
    rpc_msg_lane_fence_req req = { 0, sent };
    return send_rpc_cmd(sock, RPC_CMD_LANE_FENCE, &req, sizeof(req));
}

// Stall the consumer's main command loop until `ordinal` pushes have been
// applied there, then return. Fire-and-forget on the wire (LANE_FENCE has no
// response); the next main-socket command to that endpoint executes after the
// pushed data is in place. Call from the thread that owns the consumer's main
// socket.
GGML_RPC_SYNC_PEER_API bool ggml_backend_rpc_sync_peer_fence(
        const struct ggml_tensor * dst_probe, uint64_t ordinal) {
    const char * ep = rpc_tensor_endpoint(dst_probe);
    if (ep == nullptr) {
        return false;
    }
    auto sock = get_socket(ep);
    if (sock == nullptr) {
        return false;
    }
    rpc_msg_lane_fence_req req = { ordinal, 0 };
    return send_rpc_cmd(sock, RPC_CMD_LANE_FENCE, &req, sizeof(req));
}

// pool of events recorded on a source backend at enqueue time (i.e. in submission
// order, right after that backend's graph_compute was submitted) so a stream worker
// can wait for the source's async compute to finish before reading its output buffer.
// Without this, reading e.g. a CUDA tensor on the worker races the CUDA stream.
struct rpc_src_events {
    std::mutex mtx;
    std::unordered_map<ggml_backend_dev_t, std::vector<ggml_backend_event_t>> pool;
    static rpc_src_events & instance() { static rpc_src_events p; return p; }
};

static ggml_backend_event_t rpc_src_event_record(ggml_backend_t backend_src) {
    ggml_backend_dev_t dev = backend_src->device;
    if (dev == nullptr || backend_src->iface.event_record == nullptr) {
        return nullptr;
    }
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.events) {
        return nullptr;
    }
    auto & p = rpc_src_events::instance();
    ggml_backend_event_t ev = nullptr;
    {
        std::lock_guard<std::mutex> lock(p.mtx);
        auto & pool = p.pool[dev];
        if (!pool.empty()) {
            ev = pool.back();
            pool.pop_back();
        }
    }
    if (ev == nullptr) {
        ev = ggml_backend_event_new(dev);
        if (ev == nullptr) {
            return nullptr;
        }
    }
    ggml_backend_event_record(ev, backend_src);
    return ev;
}

static void rpc_src_event_release(ggml_backend_event_t ev) {
    auto & p = rpc_src_events::instance();
    std::lock_guard<std::mutex> lock(p.mtx);
    p.pool[ev->device].push_back(ev);
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!rpc_async_enabled()) {
        ggml_backend_rpc_buffer_set_tensor(tensor->buffer, tensor, data, offset, size);
        return;
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    // snapshot the full wire message now: the caller may reuse `data` (and the graph
    // may rewrite `tensor`) once we return
    const bool wire_bf16 = rpc_wire_bf16_ok(tensor, offset, size);
    auto msg = wire_bf16 ? rpc_prepare_set_tensor_bf16(tensor, data, offset, size)
                         : rpc_prepare_set_tensor(tensor, data, offset, size);
    const std::string endpoint = rpc_ctx->endpoint;
    const enum rpc_cmd cmd = wire_bf16 ? RPC_CMD_SET_TENSOR_BF16 : RPC_CMD_SET_TENSOR;
    rpc_ep_lanes * ep = rpc_lanes_get_active(endpoint);
    if (ep != nullptr) {
        std::lock_guard<std::mutex> l(ep->m);
        const uint64_t wait_main = ep->main_enq;
        const uint64_t wait_get  = ep->get_enq;
        ep->set_enq++;
        socket_ptr lane = ep->set_sock;
        ep->set_stream->enqueue([endpoint, lane, cmd, wait_main, wait_get, msg]{
            if (!send_lane_cmd(lane, cmd, wait_main, wait_get, msg->data(), msg->size())) {
                GGML_ABORT("[rpc fdx] SET lane to %s lost", endpoint.c_str());
            }
        });
        return;
    }
    rpc_main_enqueue_counted(endpoint, get_stream(endpoint), 1, [endpoint, msg, cmd]{
        auto sock = get_socket(endpoint);
        if (sock == nullptr) {
            GGML_LOG_ERROR("[rpc set_tensor_async] lost connection to %s\n", endpoint.c_str());
            return;
        }
        send_rpc_cmd(sock, cmd, msg->data(), msg->size());
    });
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (!rpc_async_enabled()) {
        ggml_backend_rpc_buffer_get_tensor(tensor->buffer, tensor, data, offset, size);
        return;
    }
    // route the read through the endpoint stream: the worker may be mid-send on this
    // socket (e.g. the scheduler reads MoE router ids while pipelined graphs are still
    // being submitted), so a direct socket read would corrupt the wire protocol.
    // Truly asynchronous: per the get_tensor_async contract, `data` is only guaranteed
    // valid after the caller synchronizes this backend - which drains the stream, and
    // the server's FIFO orders the read after any in-flight graph on this endpoint.
    // Blocking here instead would serialize the pipeline (e.g. the per-ubatch MTP
    // nextn-embedding read from the LAST stage would stall every prefill ubatch).
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size   = size;
    const std::string endpoint = rpc_ctx->endpoint;
    const bool wire_bf16 = rpc_wire_bf16_ok(tensor, offset, size);
    rpc_ep_lanes * ep = rpc_lanes_get_active(endpoint);
    if (ep != nullptr) {
        std::lock_guard<std::mutex> l(ep->m);
        const uint64_t wait_main = ep->main_enq;
        const uint64_t wait_set  = ep->set_enq;
        ep->get_enq++;
        socket_ptr lane = ep->get_sock;
        ep->get_stream->enqueue([endpoint, lane, request, data, size, wire_bf16, wait_main, wait_set]{
            bool ok;
            if (wire_bf16) {
                std::vector<uint8_t> wire(size / 2);
                ok = send_lane_cmd(lane, RPC_CMD_GET_TENSOR_BF16, wait_main, wait_set, &request, sizeof(request))
                  && recv_msg(lane, wire.data(), wire.size());
                if (ok) {
                    ggml_bf16_to_fp32_row((const ggml_bf16_t *) wire.data(),
                                          (float *) data, size / sizeof(float));
                }
            } else {
                ok = send_lane_cmd(lane, RPC_CMD_GET_TENSOR, wait_main, wait_set, &request, sizeof(request))
                  && recv_msg(lane, data, size);
            }
            if (!ok) {
                GGML_ABORT("[rpc fdx] GET lane to %s lost", endpoint.c_str());
            }
        });
        return;
    }
    if (wire_bf16) {
        rpc_main_enqueue_counted(endpoint, get_stream(endpoint), 1, [endpoint, request, data, size]{
            auto sock = get_socket(endpoint);
            if (sock == nullptr) {
                GGML_LOG_ERROR("[rpc get_tensor_async] lost connection to %s\n", endpoint.c_str());
                return;
            }
            std::vector<uint8_t> wire(size / 2);
            if (send_rpc_cmd(sock, RPC_CMD_GET_TENSOR_BF16, &request, sizeof(request),
                             wire.data(), wire.size())) {
                ggml_bf16_to_fp32_row((const ggml_bf16_t *) wire.data(),
                                      (float *) data, size / sizeof(float));
            }
        });
        return;
    }
    rpc_main_enqueue_counted(endpoint, get_stream(endpoint), 1, [endpoint, request, data, size]{
        auto sock = get_socket(endpoint);
        if (sock != nullptr) {
            send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
        } else {
            GGML_LOG_ERROR("[rpc get_tensor_async] lost connection to %s\n", endpoint.c_str());
        }
    });
}

static bool ggml_backend_rpc_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    if (!rpc_async_enabled()) {
        return false; // let the scheduler fall back to its synchronous copy
    }
    if (!ggml_backend_buffer_is_rpc(dst->buffer)) {
        return false; // RPC -> non-RPC copies use the scheduler's synchronous fallback
    }
    ggml_backend_rpc_context * dst_ctx = (ggml_backend_rpc_context *)backend_dst->context;
    const size_t size = ggml_nbytes(src);
    const std::string dst_endpoint = dst_ctx->endpoint;

    // [fork] direct remote->remote: hand the whole transfer to the producing
    // node so the payload never crosses the client's NIC at all
    if (rpc_peer_enabled() && ggml_backend_buffer_is_rpc(src->buffer)) {
        ggml_backend_rpc_context * src_ctx = (ggml_backend_rpc_context *)backend_src->context;
        const std::string src_endpoint = src_ctx->endpoint;
        rpc_ep_lanes * sep = src_endpoint == dst_endpoint ? nullptr : rpc_lanes_get_active(src_endpoint);
        rpc_ep_lanes * dep = sep == nullptr ? nullptr : rpc_lanes_get_active(dst_endpoint);
        if (dep != nullptr && rpc_peer_route_ready(src_endpoint, dst_endpoint, dep->session_id)) {
            rpc_msg_push_tensor_req req = {};
            // must be the address peer_open registered the link under, i.e. the
            // mapped one -- it is the key into the producer's link table
            const std::string & via = rpc_peer_addr(dst_endpoint);
            memcpy(req.endpoint, via.c_str(), via.size());
            req.src        = serialize_tensor(src);
            req.dst        = serialize_tensor(dst);
            req.src_offset = 0;
            req.dst_offset = 0;
            req.size       = size;
            req.bf16       = (rpc_wire_bf16_ok(src, 0, size) && dst->type == GGML_TYPE_F32) ? 1 : 0;
            // take the destination's SET slot: the peer's message applies in
            // the same ordered position a client-side SET would have
            {
                std::lock_guard<std::mutex> l(dep->m);
                req.wait_main = dep->main_enq;
                req.wait_get  = dep->get_enq;
                dep->set_enq++;
            }
            // ...and issue the read on the source's GET lane, after its compute
            std::lock_guard<std::mutex> l(sep->m);
            const uint64_t wait_main = sep->main_enq;
            const uint64_t wait_set  = sep->set_enq;
            sep->get_enq++;
            socket_ptr lane = sep->get_sock;
            sep->get_stream->enqueue([src_endpoint, dst_endpoint, lane, req, wait_main, wait_set]{
                rpc_msg_push_tensor_rsp rsp = { 0 };
                const bool ok = send_lane_cmd(lane, RPC_CMD_PUSH_TENSOR, wait_main, wait_set,
                                              &req, sizeof(req))
                             && recv_msg(lane, &rsp, sizeof(rsp));
                if (!ok) {
                    GGML_ABORT("[rpc peer] GET lane to %s lost during a push", src_endpoint.c_str());
                }
                if (!rsp.ok) {
                    GGML_ABORT("[rpc peer] %s could not deliver a push to %s",
                               src_endpoint.c_str(), dst_endpoint.c_str());
                }
            });
            return true;
        }
    }

    // bf16 wire: the boundary payload stays 2-byte end-to-end through the
    // star hairpin (GET_BF16 from the source lands directly in the SET_BF16
    // message) - both legs halve with no client-side conversion at all
    const bool wire_bf16 = rpc_wire_bf16_ok(src, 0, size) && dst->type == GGML_TYPE_F32;
    // snapshot the SET message header now; the payload is filled in by the tasks below
    auto msg = wire_bf16 ? rpc_prepare_set_tensor_bf16(dst, nullptr, 0, size)
                         : rpc_prepare_set_tensor(dst, nullptr, 0, size);
    const enum rpc_cmd set_cmd = wire_bf16 ? RPC_CMD_SET_TENSOR_BF16 : RPC_CMD_SET_TENSOR;

    // `prep` runs on the dst worker (lane or main stream) right before the SET
    // is sent and must leave the payload filled in at msg + RPC_SET_TENSOR_HDR
    std::function<void()> prep;

    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // RPC -> RPC (star topology): read on the source's GET lane (or its
        // main stream when lanes are off - either way the fences/FIFO order the
        // read after the source's compute), hand the payload to the dst worker
        ggml_backend_rpc_context * src_ctx = (ggml_backend_rpc_context *)backend_src->context;
        const std::string src_endpoint = src_ctx->endpoint;
        rpc_msg_get_tensor_req get_req;
        get_req.tensor = serialize_tensor(src);
        get_req.offset = 0;
        get_req.size   = size;
        auto gate = std::make_shared<rpc_gate>();
        const size_t wire_size = wire_bf16 ? size / 2 : size;
        rpc_ep_lanes * sep = rpc_lanes_get_active(src_endpoint);
        if (sep != nullptr) {
            std::lock_guard<std::mutex> l(sep->m);
            const uint64_t wait_main = sep->main_enq;
            const uint64_t wait_set  = sep->set_enq;
            sep->get_enq++;
            socket_ptr lane = sep->get_sock;
            sep->get_stream->enqueue([src_endpoint, lane, get_req, msg, wire_size, gate, wire_bf16, wait_main, wait_set]{
                const enum rpc_cmd get_cmd = wire_bf16 ? RPC_CMD_GET_TENSOR_BF16 : RPC_CMD_GET_TENSOR;
                bool ok = send_lane_cmd(lane, get_cmd, wait_main, wait_set, &get_req, sizeof(get_req))
                       && recv_msg(lane, msg->data() + RPC_SET_TENSOR_HDR, wire_size);
                if (!ok) {
                    GGML_ABORT("[rpc fdx] GET lane to %s lost", src_endpoint.c_str());
                }
                gate->set();
            });
        } else {
            rpc_main_enqueue_counted(src_endpoint, get_stream(src_endpoint), 1,
                                     [src_endpoint, get_req, msg, wire_size, gate, wire_bf16]{
                auto sock = get_socket(src_endpoint);
                if (sock != nullptr) {
                    send_rpc_cmd(sock, wire_bf16 ? RPC_CMD_GET_TENSOR_BF16 : RPC_CMD_GET_TENSOR,
                                 &get_req, sizeof(get_req), msg->data() + RPC_SET_TENSOR_HDR, wire_size);
                } else {
                    GGML_LOG_ERROR("[rpc cpy_tensor_async] lost connection to %s\n", src_endpoint.c_str());
                }
                gate->set();
            });
        }
        prep = [gate]{ gate->wait(); };
    } else if (wire_bf16) {
        // host-visible source with bf16 wire: stage the f32 read separately,
        // truncate to bf16 on the dst worker after the source compute is done
        auto staging = std::make_shared<std::vector<uint8_t>>(size);
        ggml_backend_tensor_get_async(backend_src, src, staging->data(), 0, size);
        ggml_backend_event_t ev = rpc_src_event_record(backend_src);
        prep = [msg, staging, size, ev, backend_src]() {
            if (ev != nullptr) {
                ggml_backend_event_synchronize(ev);
                rpc_src_event_release(ev);
            } else if (backend_src->iface.synchronize != nullptr) {
                ggml_backend_synchronize(backend_src);
            }
            ggml_fp32_to_bf16_row((const float *) staging->data(),
                                  (ggml_bf16_t *) (msg->data() + RPC_SET_TENSOR_HDR),
                                  size / sizeof(float));
        };
    } else {
        // host-visible source (CUDA/CPU): the payload must be read on the SOURCE
        // backend's ordered timeline. Reading it later (from the dst stream's
        // thread, gated only on a completion event) is racy whenever another
        // graph is submitted to the source backend in the meantime: under
        // pipelined prefill (n_copies > 1) the next ubatch's graph reuses the
        // same compute-buffer address and overwrites the boundary tensor,
        // tearing the copy mid-payload (measured: warm pipelined prefill
        // nondeterminism with a CUDA first stage). Enqueue the D2H on the
        // source stream NOW - submission order puts it before any later
        // graph's kernels - then let the dst worker wait until the source has
        // actually produced it before shipping.
        ggml_backend_tensor_get_async(backend_src, src, msg->data() + RPC_SET_TENSOR_HDR, 0, size);
        ggml_backend_event_t ev = rpc_src_event_record(backend_src);
        prep = [ev, backend_src]() {
            if (ev != nullptr) {
                ggml_backend_event_synchronize(ev);
                rpc_src_event_release(ev);
            } else if (backend_src->iface.synchronize != nullptr) {
                ggml_backend_synchronize(backend_src);
            }
        };
    }

    rpc_ep_lanes * dep = rpc_lanes_get_active(dst_endpoint);
    if (dep != nullptr) {
        std::lock_guard<std::mutex> l(dep->m);
        const uint64_t wait_main = dep->main_enq;
        const uint64_t wait_get  = dep->get_enq;
        dep->set_enq++;
        socket_ptr lane = dep->set_sock;
        dep->set_stream->enqueue([dst_endpoint, lane, set_cmd, wait_main, wait_get, msg, prep = std::move(prep)]{
            prep();
            if (!send_lane_cmd(lane, set_cmd, wait_main, wait_get, msg->data(), msg->size())) {
                GGML_ABORT("[rpc fdx] SET lane to %s lost", dst_endpoint.c_str());
            }
        });
    } else {
        rpc_main_enqueue_counted(dst_endpoint, get_stream(dst_endpoint), 1,
                                 [dst_endpoint, set_cmd, msg, prep = std::move(prep)]{
            prep();
            auto sock = get_socket(dst_endpoint);
            if (sock == nullptr) {
                GGML_LOG_ERROR("[rpc cpy_tensor_async] lost connection to %s\n", dst_endpoint.c_str());
                return;
            }
            send_rpc_cmd(sock, set_cmd, msg->data(), msg->size());
        });
    }
    return true;
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_event * ev = (rpc_event *)event->context;
    uint64_t g = ev->record();
    get_stream(rpc_ctx->endpoint)->enqueue([ev, g]{ ev->complete(g); });
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_event * ev = (rpc_event *)event->context;
    uint64_t target = ev->peek();
    get_stream(rpc_ctx->endpoint)->enqueue([ev, target]{ ev->wait_for(target); });
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_rpc_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(endpoint, sock, device);
    size_t max_size = get_max_size(endpoint, sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint       = */ endpoint,
        /* .device         = */ device,
        /* .name           = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(
        const std::string & endpoint, const std::shared_ptr<socket_t> & sock,
        uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd_ordered(endpoint, sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(endpoint, sock, device, free, total);
}

// ---------------------------------------------------------------------------
// Fleet hibernation: detach / reattach  [fork]
//
// Suspending the RPC hosts to disk means the client connections have to go
// away: they cannot be relied on to survive minutes or hours of the far end
// being frozen. But a disconnect is also how the daemon learns to destroy the
// session, and the session owns the KV cache. So the teardown is explicit -
// SESSION_DETACH asks the server to park its buffers and hands back a token,
// and the reattach presents that token on a fresh connection.
//
// Everything torn down here is per-connection state: the command socket, the
// async stream worker, the full-duplex lanes and their counters, the peer
// route caches. What stays allocated on the far side is exactly the memory
// that is expensive to rebuild - the KV cache, the compute buffers, and any
// weights the caller chose not to unload first.
//
// Callers must quiesce first. Nothing here interrupts an in-flight graph; it
// assumes there is none.
// ---------------------------------------------------------------------------

static std::mutex g_hib_m;
static std::unordered_map<std::string, uint64_t> g_resume_tokens;

// close the transfer lanes and reset every counter that is scoped to a server
// session, so a later reattach negotiates fresh lanes against the new session
static void rpc_lanes_teardown(const std::string & endpoint) {
    rpc_ep_lanes * ep = nullptr;
    {
        std::lock_guard<std::mutex> l(g_lanes_m);
        auto it = g_lanes.find(endpoint);
        if (it == g_lanes.end()) {
            return;
        }
        ep = it->second.get();
    }
    socket_ptr set_sock, get_sock;
    rpc_stream * set_stream = nullptr;
    rpc_stream * get_stream = nullptr;
    {
        std::lock_guard<std::mutex> l(ep->m);
        set_sock.swap(ep->set_sock);
        get_sock.swap(ep->get_sock);
        set_stream = ep->set_stream;
        get_stream = ep->get_stream;
        ep->set_stream = nullptr;
        ep->get_stream = nullptr;
        ep->main_enq = ep->set_enq = ep->get_enq = 0;
        ep->fenced_set = ep->fenced_get = 0;
        ep->barrier_set = ep->barrier_get = 0;
        ep->session_id = 0;
        ep->state = 0;   // untried: a reattach brings the lanes back up
    }
    // the workers own the sockets while they run, so stop them first
    delete set_stream;
    delete get_stream;
    if (set_sock != nullptr) {
        set_sock->shutdown_rw();
    }
    if (get_sock != nullptr) {
        get_sock->shutdown_rw();
    }
}

// peer routes name a (source endpoint, destination session) pair; both are
// gone once every session is parked, and a stale route degrades silently
static void rpc_client_reset_peer_state() {
    {
        std::lock_guard<std::mutex> l(g_peer_route_m);
        g_peer_routes.clear();
    }
    auto & st = rpc_sync_peer();
    std::lock_guard<std::mutex> l(st.m);
    st.session_ids.clear();
    st.push_lanes.clear();
    st.push_counts.clear();
    st.lane_sent.clear();
    st.lane_acked.clear();
}

static void rpc_drop_socket(const std::string & endpoint) {
    // move the stream out under the lock and destroy it outside: the
    // destructor joins the worker, and a worker task that reached for
    // get_stream() would deadlock against a held g_streams_m
    std::unique_ptr<rpc_stream> stream;
    {
        std::lock_guard<std::mutex> l(g_streams_m);
        auto it = g_streams.find(endpoint);
        if (it != g_streams.end()) {
            stream = std::move(it->second);
            g_streams.erase(it);
        }
    }
    stream.reset();
    std::lock_guard<std::mutex> l(g_sockets_m);
    g_sockets.erase(endpoint);
}

// Park every connected endpoint's session and close the connections.
// Returns the number of endpoints parked, or -1 if any of them refused.
extern "C" {

GGML_BACKEND_API int ggml_backend_rpc_detach(void) {
    std::lock_guard<std::mutex> hl(g_hib_m);
    if (g_rpc_detached.load()) {
        return g_rpc_session_lost.load() ? -1 : (int) g_resume_tokens.size();
    }

    g_resume_tokens.clear();
    g_rpc_session_lost.store(false);

    std::vector<std::pair<std::string, socket_ptr>> live;
    std::vector<std::string> missing;
    {
        std::lock_guard<std::mutex> l(g_sockets_m);
        for (auto & kv : g_sockets) {
            live.emplace_back(kv.first, kv.second);
        }
        for (const auto & endpoint : g_endpoints_seen) {
            if (g_sockets.find(endpoint) == g_sockets.end()) {
                missing.push_back(endpoint);
            }
        }
    }

    int parked  = 0;
    int refused = (int) missing.size();
    for (const auto & endpoint : missing) {
        GGML_LOG_ERROR("[rpc hibernate] %s: no live session to park\n", endpoint.c_str());
    }
    if (live.empty() && refused == 0) {
        GGML_LOG_ERROR("[rpc hibernate] detach: no endpoint is connected\n");
        refused = 1;
    }

    for (auto & item : live) {
        const std::string & endpoint = item.first;
        const socket_ptr  & sock     = item.second;

        rpc_stream * st = nullptr;
        {
            std::lock_guard<std::mutex> l(g_streams_m);
            auto it = g_streams.find(endpoint);
            st = it == g_streams.end() ? nullptr : it->second.get();
        }
        if (st != nullptr) {
            st->drain();
        }
        rpc_lanes_teardown(endpoint);

        const int patch = (int) rpc_server_patch(endpoint);
        if (patch < GGML_RPC_HIBERNATE_MIN_PATCH) {
            GGML_LOG_ERROR("[rpc hibernate] %s: server patch %d cannot park a session "
                           "(need %d) - its buffers will be lost\n",
                           endpoint.c_str(), patch, GGML_RPC_HIBERNATE_MIN_PATCH);
            refused++;
        } else {
            rpc_msg_session_detach_rsp rsp = {};
            if (!send_rpc_cmd(sock, RPC_CMD_SESSION_DETACH, nullptr, 0, &rsp, sizeof(rsp)) || rsp.token == 0) {
                GGML_LOG_ERROR("[rpc hibernate] %s: SESSION_DETACH failed\n", endpoint.c_str());
                refused++;
            } else {
                g_resume_tokens[endpoint] = rsp.token;
                parked++;
                GGML_LOG_INFO("[rpc hibernate] %s: parked %llu buffers (token %llu)\n",
                              endpoint.c_str(),
                              (unsigned long long) rsp.n_buffers,
                              (unsigned long long) rsp.token);
            }
        }
        rpc_drop_socket(endpoint);
    }

    rpc_client_reset_peer_state();
    g_rpc_session_lost.store(refused > 0);
    g_rpc_detached.store(true);
    GGML_LOG_INFO("[rpc hibernate] detached: %d parked, %d refused\n", parked, refused);
    return refused > 0 ? -refused : parked;
}

// Reconnect to every parked endpoint and re-adopt its session. A successful
// endpoint is removed from g_resume_tokens immediately, so a later retry only
// touches endpoints that are still parked.
GGML_BACKEND_API int ggml_backend_rpc_reattach(int timeout_ms) {
    std::lock_guard<std::mutex> hl(g_hib_m);
    if (g_rpc_session_lost.load()) {
        return -1;
    }
    if (!g_rpc_detached.load()) {
        return 0;
    }

    std::vector<std::string> pending;
    for (auto & kv : g_resume_tokens) {
        pending.push_back(kv.first);
    }
    if (pending.empty()) {
        g_rpc_detached.store(false);
        return 0;
    }

    g_rpc_detached.store(false);

    const int64_t deadline_us = ggml_time_us() + (int64_t) (timeout_ms > 0 ? timeout_ms : 0) * 1000;
    int  resumed   = 0;
    int  lost      = 0;
    bool announced = false;
    while (!pending.empty()) {
        std::vector<std::string> still;
        std::vector<std::string> done;
        for (const auto & endpoint : pending) {
            auto sock = get_socket(endpoint);
            if (sock == nullptr) {
                still.push_back(endpoint);
                continue;
            }
            rpc_msg_session_resume_req req = { g_resume_tokens.at(endpoint) };
            rpc_msg_session_resume_rsp rsp = {};
            if (!send_rpc_cmd(sock, RPC_CMD_SESSION_RESUME, &req, sizeof(req), &rsp, sizeof(rsp))) {
                rpc_drop_socket(endpoint);
                still.push_back(endpoint);
                continue;
            }
            if (!rsp.ok) {
                GGML_LOG_ERROR("[rpc hibernate] %s: the parked session is gone, "
                               "its buffers cannot be recovered\n", endpoint.c_str());
                rpc_drop_socket(endpoint);
                lost++;
                continue;
            }
            GGML_LOG_INFO("[rpc hibernate] %s: resumed %llu buffers\n",
                          endpoint.c_str(), (unsigned long long) rsp.n_buffers);
            // SESSION_RESUME is a main-connection command, and the server
            // bumps its ordering counter for every one of those. The client's
            // mirror (main_enq, zeroed by rpc_lanes_teardown) therefore has to
            // count it too. Miss it and the server's count stays one ahead
            // forever: every later lane fence names a target it has already
            // passed, so full-duplex SET/GET traffic is released one main
            // command early and lands out of order. Nothing errors - the
            // model just computes on tensors that arrived too soon.
            {
                rpc_ep_lanes * ep = get_ep_lanes(endpoint);
                std::lock_guard<std::mutex> el(ep->m);
                ep->main_enq++;
            }
            done.push_back(endpoint);
            resumed++;
        }
        for (const auto & endpoint : done) {
            g_resume_tokens.erase(endpoint);
        }
        pending.swap(still);
        if (lost > 0 || pending.empty() || ggml_time_us() >= deadline_us) {
            break;
        }
        if (!announced) {
            GGML_LOG_INFO("[rpc hibernate] waiting for %d endpoint(s) to come back\n", (int) pending.size());
            announced = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    if (lost > 0) {
        g_rpc_session_lost.store(true);
        g_rpc_detached.store(true);
        return -(lost + (int) pending.size());
    }
    if (!pending.empty()) {
        for (const auto & endpoint : pending) {
            GGML_LOG_ERROR("[rpc hibernate] %s: still unreachable\n", endpoint.c_str());
        }
        g_rpc_detached.store(true);
        return -(int) pending.size();
    }
    GGML_LOG_INFO("[rpc hibernate] reattached %d endpoint(s)\n", resumed);
    return resumed;
}

GGML_BACKEND_API bool ggml_backend_rpc_is_detached(void) {
    return g_rpc_detached.load();
}

GGML_BACKEND_API bool ggml_backend_rpc_session_lost(void) {
    return g_rpc_session_lost.load();
}

// Snapshot of every endpoint this process has talked to. Fills out_names and
// out_connected up to max entries and returns the total, so one probing call
// is enough to size the caller's buffers.
GGML_BACKEND_API int ggml_backend_rpc_endpoint_status(const char ** out_names, int * out_connected, int max) {
    std::lock_guard<std::mutex> l(g_sockets_m);
    const int total = (int) g_endpoints_seen.size();
    for (int i = 0; i < total && i < max; i++) {
        if (out_names != nullptr) {
            out_names[i] = g_endpoints_seen[i].c_str();
        }
        if (out_connected != nullptr) {
            out_connected[i] = g_sockets.count(g_endpoints_seen[i]) > 0 ? 1 : 0;
        }
    }
    return total;
}

} // extern "C"

// RPC server-side implementation

// Evict least recently used cache entries until the directory fits within
// limit bytes. Entry mtimes double as the LRU clock: get_cached_file bumps
// the mtime of every entry it serves.
static void rpc_cache_enforce_limit(const char * cache_dir, size_t limit) {
    if (cache_dir == nullptr || limit == 0) {
        return;
    }
    struct cache_entry {
        fs::path                path;
        size_t                  size;
        fs::file_time_type      mtime;
    };
    std::vector<cache_entry> entries;
    size_t total = 0;
    std::error_code ec;
    for (const auto & it : fs::directory_iterator(cache_dir, ec)) {
        if (!it.is_regular_file(ec)) {
            continue;
        }
        size_t size = it.file_size(ec);
        if (ec) {
            continue;
        }
        auto mtime = it.last_write_time(ec);
        if (ec) {
            continue;
        }
        entries.push_back({it.path(), size, mtime});
        total += size;
    }
    if (total <= limit) {
        return;
    }
    std::sort(entries.begin(), entries.end(), [](const cache_entry & a, const cache_entry & b) {
        return a.mtime < b.mtime;
    });
    for (const auto & entry : entries) {
        if (total <= limit) {
            break;
        }
        if (fs::remove(entry.path, ec)) {
            total -= entry.size;
            GGML_LOG_INFO("[%s] evicted '%s' (%zu bytes)\n", __func__, entry.path.string().c_str(), entry.size);
        }
    }
}

// ---------------------------------------------------------------------------
// peer links (direct remote->remote transfer) [fork]
//
// A stage boundary in the RPC star normally hairpins: the client GETs the
// boundary tensor off the producing node and SETs it onto the consuming one,
// so every activation crosses the client's NIC twice. With a peer link the
// producing server holds its own connection to the consumer -- attached as a
// RPC_LANE_PEER lane on the consumer's session, so it feeds the very same
// ordered SET queue the client's own SET lane feeds -- and the client only
// sends a ~300 byte PUSH_TENSOR instead of the payload.
//
// One link per destination endpoint, shared by every push to it; the mutex
// serializes whole messages onto the socket.
// ---------------------------------------------------------------------------
// A push is handed to a per-link sender thread and acknowledged immediately.
// It must NEVER block on the wire: the push arrives on the producer's GET
// lane, so blocking there stalls the client too, and the consumer often cannot
// accept the payload until the client has advanced it -- the client waits on
// the producer, the producer waits on the consumer, the consumer waits on the
// client. That three-way cycle deadlocked the whole fleet on the first try
// (81 MB stuck unread in one peer socket, every board's set-executor parked in
// wait_counts). Handing the bytes to a private thread breaks it: client
// progress no longer depends on peer progress, so the consumer always reaches
// the fence that lets it drain.
//
// The queue is therefore a SOFT cap -- over it we warn, never block. Depth is
// bounded in practice by the scheduler's in-flight copies (GGML_SCHED_COPIES).
struct rpc_peer_link {
    std::mutex  m;          // guards sock during connect/teardown
    socket_ptr  sock;

    std::mutex                        q_m;
    std::condition_variable           q_cv;
    std::deque<std::vector<uint8_t>>  q;      // pre-framed lane messages, FIFO
    size_t      q_bytes = 0;
    bool        closing = false;
    bool        failed  = false;
    bool        warned  = false;
    std::thread sender;
};

static size_t rpc_peer_queue_soft_cap() {
    static const size_t cap = []{
        const char * e = std::getenv("GGML_RPC_PEER_QUEUE_MB");
        long mb = e ? strtol(e, nullptr, 10) : 0;
        if (mb <= 0) {
            mb = 512;
        }
        return (size_t) mb * 1024 * 1024;
    }();
    return cap;
}

static std::mutex g_peer_mutex;
static std::unordered_map<std::string, std::shared_ptr<rpc_peer_link>> g_peer_links;

static std::shared_ptr<rpc_peer_link> rpc_peer_link_find(const std::string & endpoint) {
    std::lock_guard<std::mutex> l(g_peer_mutex);
    auto it = g_peer_links.find(endpoint);
    return it == g_peer_links.end() ? nullptr : it->second;
}

// drains the link's queue onto the socket in FIFO order
static void rpc_peer_sender(rpc_peer_link * link, std::string endpoint) {
    for (;;) {
        std::vector<uint8_t> msg;
        {
            std::unique_lock<std::mutex> l(link->q_m);
            link->q_cv.wait(l, [&]{ return !link->q.empty() || link->closing; });
            if (link->q.empty()) {
                return;   // closing and drained
            }
            msg = std::move(link->q.front());
            link->q.pop_front();
            link->q_bytes -= msg.size();
        }
        link->q_cv.notify_all();
        socket_ptr sock;
        {
            std::lock_guard<std::mutex> l(link->m);
            sock = link->sock;
        }
        if (sock == nullptr || !sock->send_data(msg.data(), msg.size())) {
            GGML_LOG_ERROR("[rpc peer] lost the peer lane to %s mid-push\n", endpoint.c_str());
            std::lock_guard<std::mutex> l(link->q_m);
            link->failed = true;
            return;
        }
    }
}

// close every peer link; called when the session that created them ends so a
// later client does not inherit sockets into a stale peer session
static void rpc_peer_links_close() {
    std::unordered_map<std::string, std::shared_ptr<rpc_peer_link>> links;
    {
        std::lock_guard<std::mutex> l(g_peer_mutex);
        links.swap(g_peer_links);
    }
    for (auto & kv : links) {
        {
            std::lock_guard<std::mutex> l(kv.second->q_m);
            kv.second->closing = true;
        }
        kv.second->q_cv.notify_all();
        if (kv.second->sender.joinable()) {
            kv.second->sender.join();
        }
        std::lock_guard<std::mutex> l(kv.second->m);
        if (kv.second->sock != nullptr) {
            kv.second->sock->shutdown_rw();
            kv.second->sock = nullptr;
        }
    }
}

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir, size_t cache_limit)
        : backends(std::move(all_backends)), cache_dir(cache_dir), cache_limit(cache_limit) {
        stored_graphs.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    size_t device_count() const { return backends.size(); }
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    // allow_cache=false for lane traffic (activations): skips the pending
    // SET_TENSOR_HASH bookkeeping, which belongs to the main thread only
    bool set_tensor(const std::vector<uint8_t> & input, bool allow_cache = true);
    // streaming form of set_tensor for the main command socket: the length
    // prefix is already consumed, the body is still on the wire. Bounds host
    // RAM to one chunk instead of the whole upload.
    bool set_tensor_stream(socket_ptr sock, uint64_t msg_size, bool allow_cache = true);
    bool set_tensor_bf16(const std::vector<uint8_t> & input, bool allow_cache = true);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool imatrix_sqsum(const rpc_msg_imatrix_sqsum_req & request, std::vector<uint8_t> & response);
    bool get_tensor_bf16(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool graph_forget(const rpc_msg_graph_forget_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);
    // [fork] direct remote->remote transfer
    bool peer_open(const rpc_msg_peer_open_req & request, rpc_msg_peer_open_rsp & response);
    bool push_tensor(const rpc_msg_push_tensor_req & request, rpc_msg_push_tensor_rsp & response);
    // [fork] fleet hibernation: hand this session's buffers to the parking
    // slot so the client can disconnect while the host suspends, and take
    // them back on the connection that returns
    bool session_detach(rpc_msg_session_detach_rsp & response);
    bool session_resume(const rpc_msg_session_resume_req & request, rpc_msg_session_resume_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph = nullptr;
    };

private:
    // Open a cache entry and report its size WITHOUT reading it. The payload is
    // streamed into the tensor in chunks by set_tensor_hash; materialising a whole
    // entry here is what used to drive a 16 GiB UMA board into its memory guard
    // (llama.cpp hands the model over in ~1 GiB buffer chunks, so entries are that
    // big). Returns false on a miss, leaving `ifs` unopened.
    bool open_cached_file(uint64_t hash, std::ifstream & ifs, size_t & size);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    // staging for open_cached_file -> ggml_backend_tensor_set. Reused so a warm
    // load does not churn a fresh chunk-sized allocation per tensor. Only the
    // main command thread runs SET_TENSOR_HASH (lane traffic takes set_tensor
    // with allow_cache=false), so this needs no locking.
    std::vector<uint8_t> cache_read_buf;
    // staging for the streaming SET_TENSOR receive. Main command thread only,
    // same as cache_read_buf, so it needs no locking.
    std::vector<uint8_t> set_recv_buf;
    size_t cache_limit;
    // set on a SET_TENSOR_HASH cache miss; the client's follow-up SET_TENSOR
    // for the same tensor region is the only upload that gets cached. Clients
    // that never offer hashes never grow the cache.
    bool       cache_pending = false;
    rpc_tensor cache_pending_tensor = {};
    uint64_t   cache_pending_offset = 0;
    uint64_t   cache_pending_hash   = 0;
    size_t     cache_hits           = 0;
    size_t     cache_misses         = 0;
    size_t     cache_hit_bytes      = 0;
    size_t     cache_upload_bytes   = 0;
    // guards `buffers`: the full-duplex lane threads deserialize tensors (and
    // thus validate buffer handles) concurrently with main-thread buffer
    // lifecycle commands. The cache_* fields stay main-thread-only.
    std::mutex buffers_mtx;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // [fork, PipeDec] deserialized graphs kept per backend, keyed by the
    // client's graph uid. The client mirrors this set with a bounded LRU and
    // evicts explicitly via GRAPH_FORGET, so lookups on RECOMPUTE never miss.
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
};

// ---------------------------------------------------------------------------
// Parked sessions  [fork, fleet hibernation]
//
// A session normally dies with its connection, and rpc_server's destructor
// frees every buffer it owns. That is the right default - a client that
// vanished is not coming back for its KV cache - but it is exactly wrong when
// the client is deliberately disconnecting so this host can suspend to disk.
//
// SESSION_DETACH therefore moves the durable half of the session (the buffer
// handle set and the deserialized graph cache) out of the rpc_server and into
// one process-wide parking slot, leaving the server object with nothing to
// free. SESSION_RESUME on a later connection moves it back. The device memory
// is never touched, so the client's remote pointers stay valid across the
// whole cycle - which is the point: those pointers are the KV cache.
//
// Only one slot exists because rpc-server serves one client at a time. If some
// other client connects and does not present the token, the park is discarded
// and its buffers freed: the new client needs the memory, and the old one has
// already lost its right to it.
// ---------------------------------------------------------------------------

struct rpc_parked_session {
    uint64_t token = 0;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    std::vector<std::unordered_map<uint64_t, rpc_server::stored_graph>> stored_graphs;
};

static std::mutex        g_parked_m;
static rpc_parked_session g_parked;

static void rpc_parked_discard_locked() {
    if (g_parked.token == 0) {
        return;
    }
    GGML_LOG_INFO("[rpc hibernate] discarding parked session %llu (%zu buffers): "
                  "another client took the device\n",
                  (unsigned long long) g_parked.token, g_parked.buffers.size());
    for (auto buffer : g_parked.buffers) {
        ggml_backend_buffer_free(buffer);
    }
    g_parked.buffers.clear();
    g_parked.stored_graphs.clear();
    g_parked.token = 0;
}

static void rpc_parked_discard() {
    std::lock_guard<std::mutex> l(g_parked_m);
    rpc_parked_discard_locked();
}

bool rpc_server::session_detach(rpc_msg_session_detach_rsp & response) {
    static std::atomic<uint64_t> next_token{1};

    std::lock_guard<std::mutex> pl(g_parked_m);
    rpc_parked_discard_locked();   // a park nobody ever came back for

    std::lock_guard<std::mutex> bl(buffers_mtx);
    g_parked.token = next_token.fetch_add(1);
    g_parked.buffers.swap(buffers);              // the destructor now frees nothing
    g_parked.stored_graphs.swap(stored_graphs);
    stored_graphs.resize(backends.size());

    response.token     = g_parked.token;
    response.n_buffers = g_parked.buffers.size();
    GGML_LOG_INFO("[rpc hibernate] parked session %llu: %zu buffers held\n",
                  (unsigned long long) g_parked.token, g_parked.buffers.size());
    return true;
}

bool rpc_server::session_resume(const rpc_msg_session_resume_req & request, rpc_msg_session_resume_rsp & response) {
    response.ok        = 0;
    response.padding   = 0;
    response.n_buffers = 0;

    std::lock_guard<std::mutex> pl(g_parked_m);
    if (g_parked.token == 0 || g_parked.token != request.token) {
        GGML_LOG_ERROR("[rpc hibernate] resume refused: token %llu does not match the parked session\n",
                       (unsigned long long) request.token);
        rpc_parked_discard_locked();
        return true;   // answered, not a protocol failure - keep the connection
    }

    std::lock_guard<std::mutex> bl(buffers_mtx);
    // a resuming connection is brand new, so this set is empty; free anything
    // that somehow is not, rather than leaking it
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    buffers.clear();
    buffers.swap(g_parked.buffers);
    stored_graphs.swap(g_parked.stored_graphs);
    if (stored_graphs.size() != backends.size()) {
        stored_graphs.resize(backends.size());
    }
    g_parked.buffers.clear();
    g_parked.stored_graphs.clear();
    g_parked.token = 0;

    response.ok        = 1;
    response.n_buffers = buffers.size();
    GGML_LOG_INFO("[rpc hibernate] resumed session %llu: %zu buffers restored\n",
                  (unsigned long long) request.token, buffers.size());
    return true;
}

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        std::lock_guard<std::mutex> lock(buffers_mtx);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(buffer) == buffers.end()) {
            GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
            return false;
        }
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(buffer) == buffers.end()) {
            GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
            return false;
        }
        buffers.erase(buffer);
    }
    ggml_backend_buffer_free(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(buffer) == buffers.end()) {
            GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
            return false;
        }
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        return false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    if (request.tensor.data < buffer_start) {
        GGML_LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        return false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    if (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        GGML_LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        return false;
    }
    if (tensor->buffer->iface.memset_tensor == nullptr) {
        GGML_LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        return false;
    }

    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 ", value: %u\n",
            __func__, (void *) tensor->buffer, tensor->data, request.offset, request.size, request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer) {
        std::lock_guard<std::mutex> lock(buffers_mtx);
        if (buffers.find(result->buffer) == buffers.end()) {
            result->buffer = nullptr;
        }
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end;
        // reject the graph with an error instead of aborting so a bad client
        // cannot take the server down. zero-sized tensors are exempt: the meta
        // backend (split-mode tensor) emits zero-sized slice views whose data
        // pointer can land past the buffer end, and they never get read
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        if (tensor_size > 0 &&
            (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size)) {
            GGML_LOG_ERROR("[%s] tensor out of buffer bounds: name=%s op=%s type=%s "
                           "ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] nb=[%zu,%zu,%zu,%zu] "
                           "data_offset=%" PRIu64 " tensor_size=%" PRIu64 " buffer_size=%" PRIu64 "\n",
                           __func__, tensor->name, ggml_op_name((ggml_op) tensor->op),
                           ggml_type_name((ggml_type) tensor->type),
                           result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                           (size_t) result->nb[0], (size_t) result->nb[1], (size_t) result->nb[2], (size_t) result->nb[3],
                           tensor->data - buffer_start, tensor_size, buffer_size);
            return nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input, bool allow_cache) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    // only cache uploads the client offered a hash for first (see cache_pending)
    const bool cache_this = allow_cache && cache_pending
        && memcmp(&cache_pending_tensor, in_tensor, sizeof(rpc_tensor)) == 0
        && cache_pending_offset == offset;
    if (allow_cache) {
        cache_pending = false;
    }
    if (cache_dir && cache_this && size > HASH_THRESHOLD) {
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, cache_pending_hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        if (rpc_cache_write_file(cache_file, data, size)) {
            GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
            rpc_cache_enforce_limit(cache_dir, cache_limit);
            cache_upload_bytes += size;
        } else {
            GGML_LOG_ERROR("[%s] failed to write cache entry '%s'\n", __func__, cache_file.string().c_str());
        }
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::open_cached_file(uint64_t hash, std::ifstream & ifs, size_t & size) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    ifs.open(cache_file, std::ios::binary);
    if (!ifs) {
        GGML_LOG_ERROR("[%s] cache entry '%s' exists but could not be opened\n",
                       __func__, cache_file.string().c_str());
        return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff end = ifs.tellg();
    if (end < 0) {
        GGML_LOG_ERROR("[%s] cache entry '%s' has no readable size\n",
                       __func__, cache_file.string().c_str());
        ifs.close();
        return false;
    }
    size = (size_t) end;
    ifs.seekg(0, std::ios::beg);
    // bump the mtime so LRU eviction sees this entry as recently used
    fs::last_write_time(cache_file, fs::file_time_type::clock::now(), ec);
    return true;
}

// How much of a cache entry is held in host RAM at once while it is copied into
// the device buffer. GGML_RPC_CACHE_CHUNK_MIB overrides it; 0 restores the old
// read-it-all behaviour. Keep this well under the smallest node's spare RAM: the
// BC-250s are 16 GiB UMA boards already holding a ~12.5 GiB shard when the cache
// is read, and their memory guard kills the daemon below a 64 MiB floor.
static size_t rpc_cache_chunk_bytes() {
    static const size_t bytes = []() -> size_t {
        const char * e = getenv("GGML_RPC_CACHE_CHUNK_MIB");
        if (e == nullptr) {
            return 32u*1024*1024;
        }
        const long mib = strtol(e, nullptr, 10);
        return mib > 0 ? (size_t) mib*1024*1024 : 0;
    }();
    return bytes;
}

// How much of an incoming SET_TENSOR body is held in host RAM at once while it
// is streamed into the device buffer. The old path received the whole message
// into one vector before touching the tensor, so a 1.2 GiB expert tensor needed
// 1.2 GiB of host RAM on top of the ~12.9 GiB shard a BC-250 was already
// holding -- that, not the cache write, is what put the board under its 64 MiB
// memory-guard floor and got the daemon killed mid-load. Peak is now one chunk.
// GGML_RPC_SET_CHUNK_MIB=0 restores the old receive-it-all behaviour.
static size_t rpc_set_chunk_bytes() {
    static const size_t bytes = []() -> size_t {
        const char * e = getenv("GGML_RPC_SET_CHUNK_MIB");
        if (e == nullptr) {
            return 32u*1024*1024;
        }
        const long mib = strtol(e, nullptr, 10);
        return mib > 0 ? (size_t) mib*1024*1024 : 0;
    }();
    return bytes;
}

// Writes one cache entry in bounded chunks, dropping each chunk from the page
// cache as it lands, and only publishes it under its hash name once the whole
// payload is down.
//
// Two problems this solves. First, an unchunked write is worse than an
// unchunked read: the payload is already resident somewhere, so the dirty pages
// the write creates are a SECOND full copy of it. On a BC-250 holding a
// ~12.9 GiB shard that doubled the largest expert tensor (1.2 GiB) and drove
// MemAvailable to 62 MiB, under the guard floor. fdatasync + FADV_DONTNEED per
// chunk bounds the extra to one chunk. Second, writing straight to the hash
// name means a daemon killed mid-write leaves a SHORT file under a valid hash,
// and open_cached_file trusts whatever length it finds -- silent weight
// corruption on the next load. Writing to `<hash>.tmp.<pid>` and renaming only
// on success makes a cache entry either whole or absent.
// GGML_RPC_CACHE_CHUNK_MIB=0 restores the old unchunked write.
class rpc_cache_writer {
public:
    rpc_cache_writer() = default;
    ~rpc_cache_writer() { abort(); }

    rpc_cache_writer(const rpc_cache_writer &)             = delete;
    rpc_cache_writer & operator=(const rpc_cache_writer &) = delete;

    bool active() const { return opened; }

    bool open(const char * cache_dir, uint64_t hash) {
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        final_path = fs::path(cache_dir) / hash_str;
        tmp_path   = fs::path(cache_dir) / (std::string(hash_str) + ".tmp." + std::to_string(rpc_cache_pid()));
#ifndef _WIN32
        fd = ::open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        opened = fd >= 0;
#else
        ofs.open(tmp_path, std::ios::binary | std::ios::trunc);
        opened = (bool) ofs;
#endif
        written = 0;
        return opened;
    }

    bool write(const void * data, size_t size) {
        if (!opened) {
            return false;
        }
        const size_t chunk = rpc_cache_chunk_bytes();
        const size_t step  = chunk == 0 ? size : chunk;
        for (size_t done = 0; done < size; ) {
            const size_t n = std::min(step, size - done);
#ifndef _WIN32
            size_t w = 0;
            while (w < n) {
                const ssize_t r = ::write(fd, (const char *) data + done + w, n - w);
                if (r <= 0) {
                    return false;
                }
                w += (size_t) r;
            }
            if (chunk != 0) {
                // land this chunk and drop it, so writeback cannot accumulate
                fdatasync(fd);
                posix_fadvise(fd, (off_t) (written + done), (off_t) n, POSIX_FADV_DONTNEED);
            }
#else
            ofs.write((const char *) data + done, (std::streamsize) n);
            ofs.flush();
            if (!ofs) {
                return false;
            }
#endif
            done += n;
        }
        written += size;
        return true;
    }

    // publish the entry under its hash name; the entry is visible only now
    bool commit() {
        if (!opened) {
            return false;
        }
        close_handle();
        opened = false;
        std::error_code ec;
        fs::rename(tmp_path, final_path, ec);
        if (ec) {
            fs::remove(tmp_path, ec);
            return false;
        }
        return true;
    }

    void abort() {
        if (!opened) {
            return;
        }
        close_handle();
        opened = false;
        std::error_code ec;
        fs::remove(tmp_path, ec);
    }

    size_t bytes_written() const { return written; }

private:
    static long rpc_cache_pid() {
#ifndef _WIN32
        return (long) getpid();
#else
        return (long) _getpid();
#endif
    }

    void close_handle() {
#ifndef _WIN32
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
#else
        ofs.close();
#endif
    }

    fs::path final_path;
    fs::path tmp_path;
    bool     opened  = false;
    size_t   written = 0;
#ifndef _WIN32
    int fd = -1;
#else
    std::ofstream ofs;
#endif
};

static bool rpc_cache_write_file(const fs::path & path, const void * data, size_t size) {
    // path is <cache_dir>/<hash>; recover the pieces the writer wants
    rpc_cache_writer writer;
    const std::string hash_str = path.filename().string();
    uint64_t hash = 0;
    if (sscanf(hash_str.c_str(), "%016" SCNx64, &hash) != 1) {
        return false;
    }
    if (!writer.open(path.parent_path().string().c_str(), hash)) {
        return false;
    }
    if (!writer.write(data, size)) {
        writer.abort();
        return false;
    }
    return writer.commit();
}

// Streaming SET_TENSOR: pull the body off the socket a chunk at a time and push
// each chunk straight into the device buffer, instead of materialising the whole
// upload in host RAM and then copying it. Same wire format as set_tensor() -- the
// caller has already consumed the u64 length prefix and passes it as msg_size.
//
// This is what makes a full-model offload fit on a 16 GiB UMA board: llama.cpp
// hands weights over in tensor-sized SET_TENSORs, GLM-5.3-Flash's largest expert
// tensor is 1.2 GiB, and the receive buffer landed on top of the board's ~12.9 GiB
// shard. Peak host RAM here is one chunk (32 MiB) regardless of tensor size.
bool rpc_server::set_tensor_stream(socket_ptr sock, uint64_t msg_size, bool allow_cache) {
    if (msg_size < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const size_t size  = (size_t) (msg_size - sizeof(rpc_tensor) - sizeof(uint64_t));
    const size_t chunk = rpc_set_chunk_bytes();
    if (chunk == 0 || size <= chunk) {
        // small enough to keep the single-buffer path: one recv, one tensor_set
        std::vector<uint8_t> input;
        try {
            input.resize((size_t) msg_size);
        } catch (const std::bad_alloc &) {
            GGML_LOG_ERROR("[%s] failed to allocate input buffer of size %" PRIu64 "\n", __func__, msg_size);
            return false;
        }
        if (!sock->recv_data(input.data(), input.size())) {
            return false;
        }
        return set_tensor(input, allow_cache);
    }

    rpc_tensor in_tensor;
    uint64_t   offset;
    if (!sock->recv_data(&in_tensor, sizeof(in_tensor))) {
        return false;
    }
    if (!sock->recv_data(&offset, sizeof(offset))) {
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu (streamed)\n",
            __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor.data + offset < p0 || in_tensor.data + offset >= p1 || size > (p1 - in_tensor.data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor.data, offset, size, p0, p1);
            return false;
        }
    }

    // only cache uploads the client offered a hash for first (see cache_pending)
    const bool cache_this = allow_cache && cache_pending
        && memcmp(&cache_pending_tensor, &in_tensor, sizeof(rpc_tensor)) == 0
        && cache_pending_offset == offset;
    const uint64_t hash = cache_pending_hash;
    if (allow_cache) {
        cache_pending = false;
    }

    rpc_cache_writer writer;
    if (cache_dir && cache_this && size > HASH_THRESHOLD) {
        if (!writer.open(cache_dir, hash)) {
            GGML_LOG_ERROR("[%s] could not open a cache entry for 0x%" PRIx64 "\n", __func__, hash);
        }
    }

    set_recv_buf.resize(chunk);
    for (size_t done = 0; done < size; ) {
        const size_t n = std::min(chunk, size - done);
        if (!sock->recv_data(set_recv_buf.data(), n)) {
            GGML_LOG_ERROR("[%s] short receive at %zu/%zu\n", __func__, done, size);
            writer.abort();
            return false;
        }
        ggml_backend_tensor_set(tensor, set_recv_buf.data(), offset + done, n);
        if (writer.active() && !writer.write(set_recv_buf.data(), n)) {
            // a failed cache write must never abort the load: drop the partial
            // entry and keep streaming into the tensor
            GGML_LOG_ERROR("[%s] failed writing cache entry 0x%" PRIx64 " at %zu/%zu\n", __func__, hash, done, size);
            writer.abort();
        }
        done += n;
    }

    if (writer.active()) {
        if (writer.commit()) {
            char hash_str[17];
            snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
            GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, hash_str);
            rpc_cache_enforce_limit(cache_dir, cache_limit);
            cache_upload_bytes += size;
        } else {
            GGML_LOG_ERROR("[%s] failed to publish cache entry 0x%" PRIx64 "\n", __func__, hash);
        }
    }
    return true;
}

bool rpc_server::set_tensor_bf16(const std::vector<uint8_t> & input, bool allow_cache) {
    // serialization format: | rpc_tensor | offset (8 bytes) | bf16 data (size/2 bytes) |
    // expands to f32 at the buffer edge; never used for weight loads (async-only)
    if (allow_cache) {
        cache_pending = false; // activations are never cache candidates
    }
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t wire_size = input.size() - sizeof(rpc_tensor) - sizeof(offset);
    const size_t size = wire_size * 2; // f32 bytes written to the tensor

    if (in_tensor->type != GGML_TYPE_F32 || wire_size % sizeof(ggml_bf16_t) != 0) {
        GGML_LOG_ERROR("[%s] bf16 wire payload for non-f32 tensor or odd size\n", __func__);
        return false;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const ggml_bf16_t * data = (const ggml_bf16_t *)(input.data() + sizeof(rpc_tensor) + sizeof(offset));
    std::vector<float> tmp(wire_size / sizeof(ggml_bf16_t));
    ggml_bf16_to_fp32_row(data, tmp.data(), (int64_t) tmp.size());
    ggml_backend_tensor_set(tensor, tmp.data(), offset, size);
    return true;
}

bool rpc_server::get_tensor_bf16(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    // request offset/size describe the f32 region; the response is bf16 (size/2)
    if (request.tensor.type != GGML_TYPE_F32 || request.size % sizeof(float) != 0) {
        GGML_LOG_ERROR("[%s] bf16 wire read of non-f32 tensor or odd size\n", __func__);
        return false;
    }
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    std::vector<float> tmp(request.size / sizeof(float));
    ggml_backend_tensor_get(tensor, tmp.data(), request.offset, request.size);
    response.resize(request.size / 2);
    ggml_fp32_to_bf16_row(tmp.data(), (ggml_bf16_t *) response.data(), (int64_t) tmp.size());
    return true;
}

// [fork] Attach a peer lane into another server's live session. Idempotent:
// a link that already exists for this endpoint is reused, so the client can
// call this once per (src,dst) pair without tracking server state.
bool rpc_server::peer_open(const rpc_msg_peer_open_req & request, rpc_msg_peer_open_rsp & response) {
    response.ok = 0;
    char ep[RPC_ENDPOINT_MAX];
    memcpy(ep, request.endpoint, sizeof(ep));
    ep[sizeof(ep) - 1] = '\0';
    const std::string endpoint(ep);
    if (endpoint.empty()) {
        GGML_LOG_ERROR("[%s] empty peer endpoint\n", __func__);
        return true;
    }
    std::shared_ptr<rpc_peer_link> link;
    {
        std::lock_guard<std::mutex> l(g_peer_mutex);
        auto it = g_peer_links.find(endpoint);
        if (it != g_peer_links.end() && it->second->sock != nullptr) {
            response.ok = 1;
            return true;
        }
        if (it != g_peer_links.end()) {
            link = it->second;
        } else {
            link = std::make_shared<rpc_peer_link>();
            g_peer_links[endpoint] = link;
        }
    }
    // connect outside the map lock: this dials another host and can block
    socket_ptr sock = rpc_lane_connect(endpoint, request.session_id, RPC_LANE_PEER);
    if (sock == nullptr) {
        GGML_LOG_WARN("[rpc peer] could not attach a peer lane to %s (session %" PRIu64 ")\n",
                      endpoint.c_str(), request.session_id);
        return true;   // reported as ok=0; the client falls back to the hairpin
    }
    {
        std::lock_guard<std::mutex> l(link->m);
        link->sock = sock;
    }
    // a previous sender for this link may have exited on a broken socket; its
    // thread object is still joinable and assigning over it would terminate()
    if (link->sender.joinable()) {
        {
            std::lock_guard<std::mutex> l(link->q_m);
            link->closing = true;
        }
        link->q_cv.notify_all();
        link->sender.join();
        std::lock_guard<std::mutex> l(link->q_m);
        link->closing = false;
        link->failed  = false;
        link->warned  = false;
        link->q.clear();
        link->q_bytes = 0;
    }
    link->sender = std::thread(rpc_peer_sender, link.get(), endpoint);
    GGML_LOG_INFO("[rpc peer] peer lane open to %s (session %" PRIu64 ")\n",
                  endpoint.c_str(), request.session_id);
    response.ok = 1;
    return true;
}

// [fork] Read one of our own tensors and ship it straight to a peer server as
// a SET_TENSOR, bypassing the client entirely. Returns false only on a
// protocol-level error (the caller then drops the connection); a transfer that
// could not be delivered reports ok=0 so the client can fail loudly.
bool rpc_server::push_tensor(const rpc_msg_push_tensor_req & request, rpc_msg_push_tensor_rsp & response) {
    response.ok = 0;
    char ep[RPC_ENDPOINT_MAX];
    memcpy(ep, request.endpoint, sizeof(ep));
    ep[sizeof(ep) - 1] = '\0';
    auto link = rpc_peer_link_find(ep);
    if (link == nullptr) {
        GGML_LOG_ERROR("[rpc peer] push to %s with no peer lane open\n", ep);
        return true;
    }
    const bool bf16 = request.bf16 != 0;
    if (bf16 && (request.src.type != GGML_TYPE_F32 || request.size % sizeof(float) != 0)) {
        GGML_LOG_ERROR("[rpc peer] bf16 push of a non-f32 tensor or odd size\n");
        return true;
    }

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    if (src == nullptr || src->buffer == nullptr) {
        GGML_LOG_ERROR("[rpc peer] error deserializing the source tensor\n");
        return true;
    }

    // sanitize src->data exactly as GET_TENSOR does
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(src->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(src->buffer);

        if (request.src.data + request.src_offset < p0 ||
            request.src.data + request.src_offset >= p1 ||
            request.size > (p1 - request.src.data - request.src_offset)) {
                GGML_LOG_ERROR("[rpc peer] source region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               request.src.data, request.src_offset, request.size, p0, p1);
                return true;
        }
    }

    // Frame the whole lane message up front -- | cmd | total | wait_main |
    // wait_get | rpc_tensor(dst) | dst_offset | data | -- reading the payload
    // straight into its final position (no staging copy in the f32 case) so
    // the sender thread can hand one contiguous buffer to the socket.
    const size_t   wire_size = bf16 ? request.size / 2 : request.size;
    const size_t   lane_hdr  = 1 + 3*sizeof(uint64_t);
    const uint64_t total     = 2*sizeof(uint64_t) + RPC_SET_TENSOR_HDR + wire_size;
    std::vector<uint8_t> msg(lane_hdr + RPC_SET_TENSOR_HDR + wire_size);
    msg[0] = (uint8_t) (bf16 ? RPC_CMD_SET_TENSOR_BF16 : RPC_CMD_SET_TENSOR);
    memcpy(msg.data() + 1,                     &total,             sizeof(total));
    memcpy(msg.data() + 1 + sizeof(uint64_t),  &request.wait_main, sizeof(request.wait_main));
    memcpy(msg.data() + 1 + 2*sizeof(uint64_t),&request.wait_get,  sizeof(request.wait_get));
    uint8_t * body = msg.data() + lane_hdr;
    memcpy(body,                          &request.dst,        sizeof(request.dst));
    memcpy(body + sizeof(request.dst),    &request.dst_offset, sizeof(request.dst_offset));
    if (bf16) {
        std::vector<float> tmp(request.size / sizeof(float));
        ggml_backend_tensor_get(src, tmp.data(), request.src_offset, request.size);
        ggml_fp32_to_bf16_row(tmp.data(), (ggml_bf16_t *) (body + RPC_SET_TENSOR_HDR),
                              (int64_t) tmp.size());
    } else {
        ggml_backend_tensor_get(src, body + RPC_SET_TENSOR_HDR, request.src_offset, request.size);
    }
    // hand off and acknowledge; never wait for the wire (see rpc_peer_link)
    {
        std::lock_guard<std::mutex> l(link->q_m);
        if (link->failed) {
            GGML_LOG_ERROR("[rpc peer] peer lane to %s already failed\n", ep);
            return true;
        }
        const size_t bytes = msg.size();
        link->q.push_back(std::move(msg));
        link->q_bytes += bytes;
        if (link->q_bytes > rpc_peer_queue_soft_cap() && !link->warned) {
            link->warned = true;
            GGML_LOG_WARN("[rpc peer] %s is backing up (%zu MB queued); the link is slower than the pipeline\n",
                          ep, link->q_bytes / (1024*1024));
        }
    }
    link->q_cv.notify_all();
    response.ok = 1;
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    cache_pending = false;
    std::ifstream cached_file;
    size_t size = 0;
    if (!open_cached_file(request.hash, cached_file, size)) {
        // cache miss: the client will follow up with a full SET_TENSOR for
        // this region - mark it as eligible for caching
        cache_pending        = true;
        cache_pending_tensor = request.tensor;
        cache_pending_offset = request.offset;
        cache_pending_hash   = request.hash;
        cache_misses++;
        response.result = 0;
        return true;
    }
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    // Stream the entry into the buffer instead of materialising it. Peak host
    // RAM is one chunk rather than the whole entry, which is what keeps a UMA
    // board off its memory guard during a warm load.
    const size_t chunk = rpc_cache_chunk_bytes();
    if (chunk == 0 || size <= chunk) {
        cache_read_buf.resize(size);
        if (size > 0 && !cached_file.read((char *) cache_read_buf.data(), (std::streamsize) size)) {
            GGML_LOG_ERROR("[%s] short read of cache entry 0x%" PRIx64 " (%zu bytes)\n",
                           __func__, request.hash, size);
            return false;
        }
        ggml_backend_tensor_set(tensor, cache_read_buf.data(), request.offset, size);
    } else {
        cache_read_buf.resize(chunk);
        for (size_t done = 0; done < size; ) {
            const size_t n = std::min(chunk, size - done);
            if (!cached_file.read((char *) cache_read_buf.data(), (std::streamsize) n)) {
                GGML_LOG_ERROR("[%s] short read of cache entry 0x%" PRIx64 " at %zu/%zu\n",
                               __func__, request.hash, done, size);
                return false;
            }
            ggml_backend_tensor_set(tensor, cache_read_buf.data(), request.offset + done, n);
            done += n;
        }
    }
    cache_hits++;
    cache_hit_bytes += size;
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

// [fork] sum-of-squares reduction of a matmul activation, done where the data
// already lives. llama-imatrix accumulates sum(x^2) per column (per expert on
// MUL_MAT_ID) and throws the activation away; pulling it home first costs
// ~10 GB per 512-token chunk of DSV4 on a 27-layer RPC split, and the whole
// imatrix pass is bandwidth-bound on that. The answer is [n_mat][ne0] floats
// plus n_mat counts, which does not grow with the token count -- so a bigger
// ubatch is now strictly cheaper per token.
bool rpc_server::imatrix_sqsum(const rpc_msg_imatrix_sqsum_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src1 = deserialize_tensor(ctx, &request.src1);
    if (src1 == nullptr || src1->buffer == nullptr || src1->type != GGML_TYPE_F32) {
        GGML_LOG_ERROR("[%s] error deserializing src1\n", __func__);
        return false;
    }

    // same bounds check the GET path does: the client controls these offsets
    auto in_bounds = [](const ggml_tensor * t) {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(t->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(t->buffer);
        const size_t d0 = (size_t) t->data;
        return d0 >= p0 && d0 < p1 && ggml_nbytes(t) <= (p1 - d0);
    };
    if (!in_bounds(src1)) {
        GGML_LOG_ERROR("[%s] src1 out of buffer bounds\n", __func__);
        return false;
    }

    const int64_t ne0   = src1->ne[0];
    const int64_t n_mat = (int64_t) request.n_mat;
    if (ne0 <= 0 || n_mat <= 0 || ne0*n_mat > (int64_t) 1 << 32) {
        GGML_LOG_ERROR("[%s] bad reduction shape ne0=%" PRId64 " n_mat=%" PRId64 "\n", __func__, ne0, n_mat);
        return false;
    }

    std::vector<char> src1_data((size_t) ggml_nbytes(src1));
    ggml_backend_tensor_get(src1, src1_data.data(), 0, src1_data.size());
    const char * data = src1_data.data();

    response.assign(sizeof(uint64_t)*n_mat + sizeof(float)*ne0*n_mat, 0);
    uint64_t * counts = (uint64_t *) response.data();
    float    * sums   = (float *) (response.data() + sizeof(uint64_t)*n_mat);

    if (request.has_ids) {
        ggml_tensor * ids = deserialize_tensor(ctx, &request.ids);
        if (ids == nullptr || ids->buffer == nullptr || !in_bounds(ids)) {
            GGML_LOG_ERROR("[%s] error deserializing ids\n", __func__);
            return false;
        }
        std::vector<char> ids_data((size_t) ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, ids_data.data(), 0, ids_data.size());

        const int64_t n_ids = ids->ne[0];
        if (ids->ne[1] != src1->ne[2]) {
            GGML_LOG_ERROR("[%s] ids/src1 token mismatch\n", __func__);
            return false;
        }
        // one pass over the routed slots; the client's loop scanned all
        // n_mat experts per slot to find the matches (256x more iterations)
        for (int64_t row = 0; row < src1->ne[2]; ++row) {
            for (int64_t idx = 0; idx < n_ids; ++idx) {
                const int32_t ex = *(const int32_t *)(ids_data.data() + row*ids->nb[1] + idx*ids->nb[0]);
                if (ex < 0 || ex >= n_mat) {
                    GGML_LOG_ERROR("[%s] expert id %d out of range\n", __func__, (int) ex);
                    return false;
                }
                const int64_t i11 = idx % src1->ne[1];
                const float * x = (const float *)(data + i11*src1->nb[1] + row*src1->nb[2]);
                float * acc = sums + (int64_t) ex * ne0;
                counts[ex]++;
                for (int64_t j = 0; j < ne0; ++j) {
                    acc[j] += x[j]*x[j];
                }
            }
        }
    } else {
        const int64_t s0_ne2 = request.src0_ne2 ? (int64_t) request.src0_ne2 : 1;
        const int64_t s0_ne3 = request.src0_ne3 ? (int64_t) request.src0_ne3 : 1;
        for (int64_t i3 = 0; i3 < src1->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src1->ne[2]; ++i2) {
                const int64_t mat_id = (i3 % s0_ne3)*s0_ne2 + (i2 % s0_ne2);
                if (mat_id < 0 || mat_id >= n_mat) {
                    GGML_LOG_ERROR("[%s] mat_id out of range\n", __func__);
                    return false;
                }
                float * acc = sums + mat_id*ne0;
                for (int64_t row = 0; row < src1->ne[1]; ++row) {
                    const float * x = (const float *)(data + row*src1->nb[1] + i2*src1->nb[2] + i3*src1->nb[3]);
                    for (int64_t j = 0; j < ne0; ++j) {
                        acc[j] += x[j]*x[j];
                    }
                }
            }
        }
        // dense tensors carry a single count over all rows, as the client does
        const int64_t nrows = ggml_nrows(src1);
        for (int64_t m = 0; m < n_mat; ++m) {
            counts[m] = (uint64_t) (nrows / n_mat);
        }
    }

    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

// [fork] GGML_RPC_GRAPH_OPS=1: one histogram per distinct (device, uid) graph.
// tau on the SPD split is linear in node count, so the question "which ops make
// up a stage graph, and how many are free views" is what picks the fusion
// target. Fires once per shape, on the server, where the graph actually is.
static void rpc_graph_ops_dump(uint32_t device, uint64_t uid, const ggml_cgraph * graph) {
    static const bool enabled = [] {
        const char * v = getenv("GGML_RPC_GRAPH_OPS");
        return v && atoi(v) != 0;
    }();
    if (!enabled || graph == nullptr) {
        return;
    }
    static std::mutex m;
    static std::unordered_set<std::string> seen;
    std::lock_guard<std::mutex> l(m);
    if (!seen.insert(std::to_string(device) + ":" + std::to_string(uid)).second) {
        return;
    }
    std::map<std::string, int> hist;
    int n_view = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        const ggml_tensor * n = graph->nodes[i];
        std::string name = ggml_op_name(n->op);
        if (n->op == GGML_OP_UNARY) {
            name += std::string("/") + ggml_unary_op_name(ggml_get_unary_op(n));
        } else if (n->op == GGML_OP_GLU) {
            name += std::string("/") + ggml_glu_op_name(ggml_get_glu_op(n));
        }
        hist[name]++;
        // the set every backend treats as free: no dispatch, no device work
        if (ggml_is_empty(n) || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_TRANSPOSE ||
            n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_NONE) {
            n_view++;
        }
    }
    std::vector<std::pair<std::string, int>> rows(hist.begin(), hist.end());
    std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second > b.second; });
    fprintf(stderr, "[rpc graph ops] dev=%u uid=%" PRIu64 " nodes=%d free_views=%d dispatched=%d\n",
            device, uid, graph->n_nodes, n_view, graph->n_nodes - n_view);
    for (const auto & r : rows) {
        fprintf(stderr, "[rpc graph ops]   %-28s %5d\n", r.first.c_str(), r.second);
    }
    fflush(stderr);
}

// [fork] Batched trace output.
//
// The daemons run with stdout/stderr redirected straight to a log file on the
// node's root filesystem. On the boards whose root is a USB-NVMe bridge (the
// four carrying a Mellanox card in the M.2 slot), that filesystem is btrfs on a
// JMicron/uas device, and btrfs commits its transaction every 30 s. A write()
// landing inside a commit blocks in D-state for seconds. GGML_RPC_GRAPH_TRACE
// used to fprintf+fflush once per graph -- ~72 syscalls/s per board -- so the
// daemon was nearly always inside the filesystem when a commit stalled. That
// showed up as 0.4-3.3 s mid-generation pipeline stalls, on those four boards
// only and never on the five with SATA SSDs (measured 2026-08-05).
//
// Batching keeps every line and every byte, but collapses the syscall rate by
// ~700x so the daemon is rarely in write() when a commit hits. Trace lines stay
// in order among themselves; only unrelated stderr output can interleave
// differently. GGML_RPC_GRAPH_TRACE_FLUSH_MS=0 restores line-at-a-time output.
static std::mutex  g_trace_mtx;
static std::string g_trace_buf;
static int64_t     g_trace_last_us = 0;

static void rpc_trace_flush() {
    std::lock_guard<std::mutex> lock(g_trace_mtx);
    if (!g_trace_buf.empty()) {
        fwrite(g_trace_buf.data(), 1, g_trace_buf.size(), stderr);
        fflush(stderr);
        g_trace_buf.clear();
    }
}

static void rpc_trace_emit(const char * fmt, ...) {
    static const int64_t flush_us = [] {
        const char * v = getenv("GGML_RPC_GRAPH_TRACE_FLUSH_MS");
        return (int64_t)(v ? atoi(v) : 5000) * 1000;
    }();
    static const bool registered = [] {
        std::atexit(rpc_trace_flush);
        return true;
    }();
    (void) registered;

    char line[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n <= 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_trace_mtx);
    g_trace_buf.append(line, std::min((size_t) n, sizeof(line) - 1));

    const int64_t now = ggml_time_us();
    if (g_trace_last_us == 0) {
        g_trace_last_us = now;
    }
    if (g_trace_buf.size() >= 64*1024 || now - g_trace_last_us >= flush_us) {
        fwrite(g_trace_buf.data(), 1, g_trace_buf.size(), stderr);
        fflush(stderr);
        g_trace_buf.clear();
        g_trace_last_us = now;
    }
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    static const bool graph_trace = [] {
        const char * value = getenv("GGML_RPC_GRAPH_TRACE");
        return value && atoi(value) != 0;
    }();
    const int64_t t_start = graph_trace ? ggml_time_us() : 0;
    // [fork] idle accounting: how long this device sat with nothing to compute
    // since the previous graph returned. With an absolute (epoch) stamp on
    // every line, traces from several daemons can be merged to see whether the
    // pipeline stages actually overlap or run one after another.
    static int64_t t_prev_end = 0;
    const double   t_wait_ms  = (graph_trace && t_prev_end) ? (t_start - t_prev_end) / 1000.0 : 0.0;

    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t) + sizeof(uint64_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint64_t uid;
    memcpy(&uid, src, sizeof(uid));
    src += sizeof(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    const size_t hdr_size = 2*sizeof(uint32_t) + sizeof(uint64_t);
    if (input.size() < hdr_size + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < hdr_size + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);
    const int64_t t_parse = graph_trace ? ggml_time_us() : 0;

    stored_graph sg;
    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    sg.buffer.resize(buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
        if (graph->nodes[i] != nullptr) {
            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
        }
    }
    const int64_t t_build = graph_trace ? ggml_time_us() : 0;
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    const int64_t t_compute = graph_trace ? ggml_time_us() : 0;
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    rpc_graph_ops_dump(device, uid, graph);
    if (uid != 0) {
        sg.graph = graph;
        stored_graphs[device][uid] = std::move(sg);
        if (stored_graphs[device].size() > RPC_GRAPH_CACHE_SLOTS + 8) {
            // the client's LRU + GRAPH_FORGET should keep this bounded
            GGML_LOG_WARN("[%s] device %u holds %zu cached graphs - client eviction may be broken\n",
                    __func__, device, stored_graphs[device].size());
        }
    }
    if (graph_trace) {
        const int64_t t_end = ggml_time_us();
        const double  epoch = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        rpc_trace_emit(
                "[rpc graph] t=%.3f dev=%u nodes=%u tensors=%u wait=%.2fms parse=%.2fms build=%.2fms compute=%.2fms store=%.2fms total=%.2fms\n",
                epoch, device, n_nodes, n_tensors, t_wait_ms,
                (t_parse   - t_start)   / 1000.0,
                (t_build   - t_parse)   / 1000.0,
                (t_compute - t_build)   / 1000.0,
                (t_end     - t_compute) / 1000.0,
                (t_end     - t_start)   / 1000.0);
        t_prev_end = ggml_time_us();
    }
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    // [fork] This -- not graph_compute -- is the decode hot path: once a graph
    // shape repeats, the client sends only a uid. It was the one command with no
    // timing at all, which made every steady-state number an inference. Same
    // GGML_RPC_GRAPH_TRACE gate; `compute` here is the device's honest cost for
    // the stage, so client stage time minus this is the RPC overhead.
    static const bool graph_trace = [] {
        const char * value = getenv("GGML_RPC_GRAPH_TRACE");
        return value && atoi(value) != 0;
    }();
    static int64_t t_prev_end = 0;
    const int64_t  t_start    = graph_trace ? ggml_time_us() : 0;
    const double   t_wait_ms  = (graph_trace && t_prev_end) ? (t_start - t_prev_end) / 1000.0 : 0.0;

    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        GGML_LOG_ERROR("[%s] device: %u, uid: %" PRIu64 " not in graph cache\n", __func__, device, request.uid);
        return false;
    }
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    const int64_t t_lookup = graph_trace ? ggml_time_us() : 0;
    ggml_status status = ggml_backend_graph_compute(backends[device], it->second.graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    if (graph_trace) {
        const int64_t t_end = ggml_time_us();
        const double  epoch = std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        rpc_trace_emit(
                "[rpc regraph] t=%.3f dev=%u nodes=%d wait=%.2fms lookup=%.2fms compute=%.2fms\n",
                epoch, device, it->second.graph->n_nodes, t_wait_ms,
                (t_lookup - t_start) / 1000.0,
                (t_end    - t_lookup) / 1000.0);
        t_prev_end = ggml_time_us();
    }
    return true;
}

bool rpc_server::graph_forget(const rpc_msg_graph_forget_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    stored_graphs[device].erase(request.uid);
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    if (cache_hits > 0 || cache_misses > 0) {
        GGML_LOG_INFO("[rpc_cache] hits=%zu (%.2f GiB), misses=%zu, uploaded=%.2f GiB\n",
                cache_hits, cache_hit_bytes / double(1024ull * 1024ull * 1024ull),
                cache_misses, cache_upload_bytes / double(1024ull * 1024ull * 1024ull));
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

// ---------------------------------------------------------------------------
// Full-duplex lane serving  [fork, pipeline-prefill Phase 2]
//
// A session is one main client connection plus up to two attached lane
// connections. Execution order across the three connections is reconstructed
// from the per-lane processed-command counters: every lane command carries the
// counts of the other lanes at client submission time and executes only once
// those counts are reached; the main lane orders itself after lane traffic via
// LANE_FENCE. Transfers overlap compute: the SET lane has a dedicated reader
// thread that keeps draining the wire into a bounded queue while the executor
// waits on fences, and the GET lane advances its counter before streaming the
// response back out.
// ---------------------------------------------------------------------------

struct rpc_lane_set_msg {
    uint8_t  cmd = 0;
    uint64_t wait_main = 0;
    uint64_t wait_get  = 0;
    std::vector<uint8_t> payload;
};

struct rpc_active_session {
    uint64_t     id     = 0;
    rpc_server * server = nullptr;

    // execution-order counters: commands fully processed per lane
    std::mutex              sync_m;
    std::condition_variable sync_cv;
    uint64_t main_done = 0, set_done = 0, get_done = 0;
    bool     failed    = false;

    // lane sockets/threads (attached while the session is live)
    std::mutex  lane_m;
    bool        closing = false;
    socket_ptr  set_sock, get_sock;
    std::thread set_reader_th, set_exec_th, get_th;
    // [fork] peer lanes: other servers pushing activations straight to us.
    // Each gets its own reader/applier thread (see rpc_lane_peer_serve).
    std::vector<socket_ptr>  peer_socks;
    std::vector<std::thread> peer_ths;

    // SET lane reader -> executor queue (bounded by bytes)
    std::mutex              q_m;
    std::condition_variable q_cv;
    std::deque<rpc_lane_set_msg> q;
    size_t q_bytes  = 0;
    bool   q_closed = false;

    void bump(uint64_t & ctr) {
        {
            std::lock_guard<std::mutex> l(sync_m);
            ctr++;
        }
        sync_cv.notify_all();
    }

    // wait until the counters reach the targets; false once the session failed
    bool wait_counts(uint64_t need_main, uint64_t need_set, uint64_t need_get) {
        std::unique_lock<std::mutex> l(sync_m);
        sync_cv.wait(l, [&]{
            return failed || (main_done >= need_main && set_done >= need_set && get_done >= need_get);
        });
        return !failed;
    }

    void fail() {
        {
            std::lock_guard<std::mutex> l(sync_m);
            failed = true;
        }
        sync_cv.notify_all();
        {
            std::lock_guard<std::mutex> l(q_m);
            q_closed = true;
        }
        q_cv.notify_all();
    }
};

static size_t rpc_lane_queue_cap() {
    static const size_t cap = []{
        const char * e = std::getenv("GGML_RPC_LANE_QUEUE_MB");
        long mb = e ? strtol(e, nullptr, 10) : 0;
        if (mb <= 0) {
            mb = 256;
        }
        return (size_t) mb * 1024 * 1024;
    }();
    return cap;
}

// Reads SET messages off `sock` into the session's ordered apply queue.
static void rpc_lane_set_reader(rpc_active_session * s, socket_ptr sock, bool is_peer) {
    for (;;) {
        uint8_t cmd;
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd != RPC_CMD_SET_TENSOR && cmd != RPC_CMD_SET_TENSOR_BF16) {
            GGML_LOG_ERROR("[rpc fdx] unexpected command %d on %s lane\n", cmd, is_peer ? "peer" : "SET");
            break;
        }
        uint64_t size;
        if (!sock->recv_data(&size, sizeof(size))) {
            break;
        }
        if (size < 2*sizeof(uint64_t) || size > MAX_CHUNK_SIZE) {
            GGML_LOG_ERROR("[rpc fdx] bad SET lane message size %" PRIu64 "\n", size);
            break;
        }
        rpc_lane_set_msg m;
        m.cmd = cmd;
        if (!sock->recv_data(&m.wait_main, sizeof(m.wait_main))) {
            break;
        }
        if (!sock->recv_data(&m.wait_get, sizeof(m.wait_get))) {
            break;
        }
        m.payload.resize(size - 2*sizeof(uint64_t));
        if (!sock->recv_data(m.payload.data(), m.payload.size())) {
            break;
        }
        const size_t bytes = m.payload.size();
        {
            std::unique_lock<std::mutex> l(s->q_m);
            s->q_cv.wait(l, [&]{ return s->q_closed || s->q_bytes < rpc_lane_queue_cap(); });
            if (s->q_closed) {
                return;
            }
            s->q.push_back(std::move(m));
            s->q_bytes += bytes;
        }
        s->q_cv.notify_all();
    }
    // socket error or client teardown: fail the session so fence waiters and
    // the executor cannot hang on commands that will never arrive. A peer that
    // hangs up while we are already closing is just the far end tearing down
    // its own session, so do not turn that into an error.
    if (is_peer) {
        std::lock_guard<std::mutex> l(s->lane_m);
        if (s->closing) {
            return;
        }
    }
    s->fail();
}

static void rpc_lane_set_exec(rpc_active_session * s) {
    for (;;) {
        rpc_lane_set_msg m;
        {
            std::unique_lock<std::mutex> l(s->q_m);
            s->q_cv.wait(l, [&]{ return !s->q.empty() || s->q_closed; });
            if (s->q.empty()) {
                return; // closed and drained
            }
            m = std::move(s->q.front());
            s->q.pop_front();
            s->q_bytes -= m.payload.size();
        }
        s->q_cv.notify_all();
        if (!s->wait_counts(m.wait_main, 0, m.wait_get)) {
            return;
        }
        const bool ok = m.cmd == RPC_CMD_SET_TENSOR
            ? s->server->set_tensor(m.payload, /*allow_cache =*/ false)
            : s->server->set_tensor_bf16(m.payload, /*allow_cache =*/ false);
        if (!ok) {
            GGML_LOG_ERROR("[rpc fdx] SET lane apply failed\n");
            s->fail();
            return;
        }
        s->bump(s->set_done);
    }
}

static void rpc_lane_get_serve(rpc_active_session * s) {
    socket_ptr sock = s->get_sock;
    for (;;) {
        uint8_t cmd;
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        // PUSH_TENSOR rides the GET lane because it is a local read like the
        // others -- the only difference is where the bytes go afterwards, so it
        // wants exactly the same ordering against this node's compute [fork]
        const bool is_push = cmd == RPC_CMD_PUSH_TENSOR;
        if (cmd != RPC_CMD_GET_TENSOR && cmd != RPC_CMD_GET_TENSOR_BF16 && !is_push) {
            GGML_LOG_ERROR("[rpc fdx] unexpected command %d on GET lane\n", cmd);
            break;
        }
        uint64_t size;
        if (!sock->recv_data(&size, sizeof(size))) {
            break;
        }
        const size_t expect = is_push ? sizeof(rpc_msg_push_tensor_req) : sizeof(rpc_msg_get_tensor_req);
        if (size != 2*sizeof(uint64_t) + expect) {
            GGML_LOG_ERROR("[rpc fdx] bad GET lane message size %" PRIu64 "\n", size);
            break;
        }
        uint64_t wait_main, wait_set;
        if (!sock->recv_data(&wait_main, sizeof(wait_main))) {
            break;
        }
        if (!sock->recv_data(&wait_set, sizeof(wait_set))) {
            break;
        }
        rpc_msg_get_tensor_req  req;
        rpc_msg_push_tensor_req push_req;
        if (!sock->recv_data(is_push ? (void *) &push_req : (void *) &req, expect)) {
            break;
        }
        if (!s->wait_counts(wait_main, wait_set, 0)) {
            return;
        }
        std::vector<uint8_t> response;
        bool ok;
        if (is_push) {
            // the whole payload leaves for the peer here; the client gets a
            // one-byte ack so the lane stays request/response framed
            rpc_msg_push_tensor_rsp push_rsp;
            ok = s->server->push_tensor(push_req, push_rsp);
            response.assign((const uint8_t *) &push_rsp,
                            (const uint8_t *) &push_rsp + sizeof(push_rsp));
        } else {
            ok = cmd == RPC_CMD_GET_TENSOR
                ? s->server->get_tensor(req, response)
                : s->server->get_tensor_bf16(req, response);
        }
        if (!ok) {
            GGML_LOG_ERROR("[rpc fdx] GET lane %s failed\n", is_push ? "push" : "read");
            break;
        }
        // advance the counter BEFORE streaming the response out: the read into
        // host memory is what ordering needs; the wire transfer overlaps the
        // next command's compute
        s->bump(s->get_done);
        if (!send_msg(sock, response.data(), response.size())) {
            break;
        }
    }
    s->fail();
}

// [fork] One thread per peer lane: read a SET message, wait for this node's
// ordering counters, apply it, bump set_done.
//
// Deliberately NOT sharing the client SET lane's queue and executor. A peer
// message arrives decoupled from the client's stream order -- the producer
// ships it as soon as its own compute is done, which can be long before the
// client has sent us the commands its fence targets refer to. On the shared
// queue such a message parks at the head and blocks every client upload behind
// it, including the ones that would advance the very counters it waits for
// (measured: 256 MB queued on one board, both readers and the executor stuck).
// With a private thread per link a not-yet-satisfiable push only stalls its own
// producer, via ordinary TCP backpressure.
//
// set_done is a count of applied SET-domain operations, so LANE_FENCE keeps
// working across the client lane and every peer lane; the boundary tensors
// these carry are distinct, so applying them out of order relative to each
// other is not observable.
static void rpc_lane_peer_serve(rpc_active_session * s, socket_ptr sock) {
    for (;;) {
        uint8_t cmd;
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd != RPC_CMD_SET_TENSOR && cmd != RPC_CMD_SET_TENSOR_BF16) {
            GGML_LOG_ERROR("[rpc peer] unexpected command %d on a peer lane\n", cmd);
            break;
        }
        uint64_t size;
        if (!sock->recv_data(&size, sizeof(size))) {
            break;
        }
        if (size < 2*sizeof(uint64_t) || size > MAX_CHUNK_SIZE) {
            GGML_LOG_ERROR("[rpc peer] bad peer lane message size %" PRIu64 "\n", size);
            break;
        }
        uint64_t wait_main, wait_get;
        if (!sock->recv_data(&wait_main, sizeof(wait_main)) ||
            !sock->recv_data(&wait_get,  sizeof(wait_get))) {
            break;
        }
        std::vector<uint8_t> payload(size - 2*sizeof(uint64_t));
        if (!sock->recv_data(payload.data(), payload.size())) {
            break;
        }
        if (!s->wait_counts(wait_main, 0, wait_get)) {
            return;
        }
        const bool ok = cmd == RPC_CMD_SET_TENSOR
            ? s->server->set_tensor(payload, /*allow_cache =*/ false)
            : s->server->set_tensor_bf16(payload, /*allow_cache =*/ false);
        if (!ok) {
            GGML_LOG_ERROR("[rpc peer] peer lane apply failed\n");
            s->fail();
            return;
        }
        s->bump(s->set_done);
    }
    // a peer hanging up while we are already closing is just the far end
    // tearing down its own session, not an error
    {
        std::lock_guard<std::mutex> l(s->lane_m);
        if (s->closing) {
            return;
        }
    }
    s->fail();
}

// registry so lane connections (classified on the acceptor's dispatch threads)
// can find the live session
static std::mutex g_session_mutex;
static rpc_active_session * g_session_active = nullptr;

static bool rpc_session_attach_lane(uint64_t session_id, uint8_t lane, socket_ptr sock) {
    std::lock_guard<std::mutex> glock(g_session_mutex);
    rpc_active_session * s = g_session_active;
    if (s == nullptr || s->id != session_id || s->server == nullptr) {
        return false;
    }
    std::lock_guard<std::mutex> l(s->lane_m);
    if (s->closing) {
        return false;
    }
    if (lane == RPC_LANE_SET && s->set_sock == nullptr) {
        s->set_sock      = sock;
        s->set_reader_th = std::thread(rpc_lane_set_reader, s, sock, /*is_peer =*/ false);
        s->set_exec_th   = std::thread(rpc_lane_set_exec, s);
        return true;
    }
    if (lane == RPC_LANE_GET && s->get_sock == nullptr) {
        s->get_sock = sock;
        s->get_th   = std::thread(rpc_lane_get_serve, s);
        return true;
    }
    if (lane == RPC_LANE_PEER) {
        s->peer_socks.push_back(sock);
        s->peer_ths.emplace_back(rpc_lane_peer_serve, s, sock);
        GGML_LOG_INFO("[rpc peer] peer lane attached to session %" PRIu64 "\n", session_id);
        return true;
    }
    return false;
}

static void rpc_session_shutdown_lanes(rpc_active_session & s) {
    {
        std::lock_guard<std::mutex> l(s.lane_m);
        s.closing = true;
        if (s.set_sock != nullptr) {
            s.set_sock->shutdown_rw();
        }
        if (s.get_sock != nullptr) {
            s.get_sock->shutdown_rw();
        }
        for (auto & p : s.peer_socks) {
            if (p != nullptr) {
                p->shutdown_rw();
            }
        }
    }
    s.fail();   // wake any fence/queue waiter
    if (s.set_reader_th.joinable()) {
        s.set_reader_th.join();
    }
    for (auto & t : s.peer_ths) {
        if (t.joinable()) {
            t.join();
        }
    }
    if (s.set_exec_th.joinable()) {
        s.set_exec_th.join();
    }
    if (s.get_th.joinable()) {
        s.get_th.join();
    }
    // drop our outgoing links too: they belong to the session that opened them
    rpc_peer_links_close();
}

static void rpc_serve_client(rpc_server & server, rpc_active_session & session, socket_ptr sock) {
    uint8_t cmd;
    // the HELLO handshake was already completed by the connection dispatcher
    // the first command decides the fate of a parked session: anything other
    // than SESSION_RESUME means this client is not the one that parked it, and
    // holding its buffers any longer would just deny this client the memory
    bool first_cmd = true;
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        if (first_cmd) {
            switch (cmd) {
                // read-only probes: a version check or a device enumeration
                // must not cost a parked session its buffers
                case RPC_CMD_DEVICE_COUNT:
                case RPC_CMD_GET_ALIGNMENT:
                case RPC_CMD_GET_MAX_SIZE:
                case RPC_CMD_SESSION_INFO:
                    break;
                case RPC_CMD_SESSION_RESUME:
                    first_cmd = false;
                    break;
                default:
                    // anything else - including GET_DEVICE_MEMORY, which is a
                    // client about to size a split - means this connection owns
                    // the device now, and the park is denying it the memory
                    first_cmd = false;
                    rpc_parked_discard();
                    break;
            }
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = server.device_count();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.memset_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                // streamed, not recv_msg'd: weight uploads are tensor-sized and
                // must not be materialised in host RAM on a 16 GiB UMA board
                uint64_t msg_size;
                if (!sock->recv_data(&msg_size, sizeof(msg_size))) {
                    return;
                }
                if (!server.set_tensor_stream(sock, msg_size)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_BF16: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor_bf16(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_BF16: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor_bf16(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_IMATRIX_SQSUM: {
                rpc_msg_imatrix_sqsum_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.imatrix_sqsum(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_FORGET: {
                rpc_msg_graph_forget_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_forget(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SESSION_INFO: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_session_info_rsp response;
                response.session_id = session.id;
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SESSION_DETACH: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_session_detach_rsp response;
                if (!server.session_detach(response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SESSION_RESUME: {
                rpc_msg_session_resume_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_session_resume_rsp response;
                if (!server.session_resume(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_LANE_FENCE: {
                rpc_msg_lane_fence_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!session.wait_counts(0, request.wait_set, request.wait_get)) {
                    return;
                }
                break;
            }
            case RPC_CMD_LANE_ATTACH: {
                // only valid as the first command of a dedicated lane connection
                GGML_LOG_ERROR("LANE_ATTACH on the main connection\n");
                return;
            }
            case RPC_CMD_PEER_OPEN: {
                rpc_msg_peer_open_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_peer_open_rsp response;
                if (!server.peer_open(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_PUSH_TENSOR: {
                // pushes are routed on the GET lane so they order against this
                // node's compute; on the main connection they would serialize
                // behind graph traffic for no benefit
                GGML_LOG_ERROR("PUSH_TENSOR on the main connection\n");
                return;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
        // every processed main-lane command advances the ordering counter the
        // lane fences wait on (the client mirrors this count at submission)
        session.bump(session.main_done);
    }
}

// Completes the HELLO handshake on a fresh connection (the cmd byte was
// already consumed by the connection dispatcher).
static bool rpc_handshake_hello(const socket_ptr & sock) {
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return false;
    }
    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return false;
    }
    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return false;
    }
    rpc_msg_hello_rsp rsp = {};
    rsp.major = RPC_PROTO_MAJOR_VERSION;
    rsp.minor = RPC_PROTO_MINOR_VERSION;
    rsp.patch = RPC_PROTO_PATCH_VERSION;
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return false;
    }
    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    return true;
}

// accepted-and-handshaked main connections waiting to be served
struct rpc_pending_conns {
    std::mutex              m;
    std::condition_variable cv;
    std::deque<socket_ptr>  q;
    bool done = false;
};

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir, size_t cache_limit,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    if (cache_dir && cache_limit > 0) {
        printf("  cache limit    : %zu MiB\n", cache_limit / (1024 * 1024));
        rpc_cache_enforce_limit(cache_dir, cache_limit);
    }
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    // Connections are accepted and classified on a dedicated thread so that
    // full-duplex lane connections can attach while the main thread is busy
    // serving a session. Each fresh connection is classified on a short-lived
    // thread of its own: a connected-but-silent peer must not stall accepts.
    auto pending = std::make_shared<rpc_pending_conns>();
    std::thread acceptor([server_socket, pending]{
        for (;;) {
            auto conn = server_socket->accept();
            if (conn == nullptr) {
                std::lock_guard<std::mutex> l(pending->m);
                pending->done = true;
                pending->cv.notify_all();
                return;
            }
            std::thread([conn, pending]{
                uint8_t cmd = 0;
                if (!conn->recv_data(&cmd, 1)) {
                    return;
                }
                if (cmd == RPC_CMD_HELLO) {
                    if (!rpc_handshake_hello(conn)) {
                        return;
                    }
                    {
                        std::lock_guard<std::mutex> l(pending->m);
                        pending->q.push_back(conn);
                    }
                    pending->cv.notify_all();
                } else if (cmd == RPC_CMD_LANE_ATTACH) {
                    uint64_t sz;
                    if (!conn->recv_data(&sz, sizeof(sz)) || sz != sizeof(rpc_msg_lane_attach_req)) {
                        return;
                    }
                    rpc_msg_lane_attach_req req;
                    if (!conn->recv_data(&req, sizeof(req))) {
                        return;
                    }
                    rpc_msg_lane_attach_rsp rsp;
                    rsp.ok = rpc_session_attach_lane(req.session_id, req.lane, conn) ? 1 : 0;
                    send_msg(conn, &rsp, sizeof(rsp));
                    // an attached socket is now owned by the session's lane
                    // threads; a rejected one is dropped here
                } else {
                    GGML_LOG_ERROR("Expected HELLO or LANE_ATTACH as first command, got %d\n", cmd);
                }
            }).detach();
        }
    });

    uint64_t next_session_id = 1;
    for (;;) {
        socket_ptr client_socket;
        {
            std::unique_lock<std::mutex> l(pending->m);
            pending->cv.wait(l, [&]{ return pending->done || !pending->q.empty(); });
            if (pending->q.empty()) {
                fprintf(stderr, "Failed to accept client connection\n");
                break;
            }
            client_socket = pending->q.front();
            pending->q.pop_front();
        }
        char ts[32];
        {
            time_t now = time(nullptr);
            strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));
        }
        std::string peer = client_socket->peer_str();
        printf("[%s] accepted connection from %s\n", ts, peer.c_str());
        fflush(stdout);
        {
            rpc_active_session session;
            session.id = next_session_id++;
            rpc_server server(backends, cache_dir, cache_limit);
            session.server = &server;
            {
                std::lock_guard<std::mutex> l(g_session_mutex);
                g_session_active = &session;
            }
            rpc_serve_client(server, session, client_socket);
            {
                std::lock_guard<std::mutex> l(g_session_mutex);
                g_session_active = nullptr;
            }
            // lane threads reference `server`; join them before it is destroyed
            rpc_session_shutdown_lanes(session);
        }
        {
            time_t now = time(nullptr);
            strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));
        }
        printf("[%s] connection from %s closed\n", ts, peer.c_str());
        fflush(stdout);
    }
    if (acceptor.joinable()) {
        acceptor.join();
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_endpoint(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;
    return ctx->endpoint.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ rpc_async_enabled(),
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ rpc_async_enabled(),
        /* .mmap_support          = */ true,
    };
}

// --- device-level event ops [fork, PipeDec Phase 1] ---

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_event_t event = new ggml_backend_event();
    event->device  = dev;
    event->context = new rpc_event();
    return event;
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    delete (rpc_event *)event->context;
    delete event;
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    rpc_event * ev = (rpc_event *)event->context;
    ev->wait_for(ev->peek());
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_set_client_cache") == 0) {
        return (void *)ggml_backend_rpc_set_client_cache;
    }
    if (std::strcmp(name, "ggml_backend_rpc_get_client_cache") == 0) {
        return (void *)ggml_backend_rpc_get_client_cache;
    }
    if (std::strcmp(name, "ggml_backend_rpc_cache_threshold") == 0) {
        return (void *)ggml_backend_rpc_cache_threshold;
    }
    if (std::strcmp(name, "ggml_backend_rpc_buffer_cache_query") == 0) {
        return (void *)ggml_backend_rpc_buffer_cache_query;
    }
    if (std::strcmp(name, "ggml_backend_rpc_buffer_cache_upload") == 0) {
        return (void *)ggml_backend_rpc_buffer_cache_upload;
    }
    if (std::strcmp(name, "ggml_backend_rpc_buffer_endpoint") == 0) {
        return (void *)ggml_backend_rpc_buffer_endpoint;
    }
    if (std::strcmp(name, "ggml_backend_rpc_device_endpoint") == 0) {
        return (void *)ggml_backend_rpc_device_get_endpoint;
    }
    if (std::strcmp(name, "ggml_backend_rpc_detach") == 0) {
        return (void *)ggml_backend_rpc_detach;
    }
    if (std::strcmp(name, "ggml_backend_rpc_reattach") == 0) {
        return (void *)ggml_backend_rpc_reattach;
    }
    if (std::strcmp(name, "ggml_backend_rpc_is_detached") == 0) {
        return (void *)ggml_backend_rpc_is_detached;
    }
    if (std::strcmp(name, "ggml_backend_rpc_session_lost") == 0) {
        return (void *)ggml_backend_rpc_session_lost;
    }
    if (std::strcmp(name, "ggml_backend_rpc_endpoint_status") == 0) {
        return (void *)ggml_backend_rpc_endpoint_status;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd_ordered(endpoint, sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
            /* .known_graph_uids = */ {},
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
