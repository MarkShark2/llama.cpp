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

static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

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

    char padding[4];
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
    RPC_CMD_GRAPH_FORGET,
    RPC_CMD_COUNT,
};

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

struct ggml_backend_rpc_buffer_context {
    std::string endpoint;
    std::shared_ptr<socket_t> sock;
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
    return sock->send_data(msg, msg_size);
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

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    rpc_cmd_stats_add(cmd, input_size);
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
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
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
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

    sock->update_caps(response.conn_caps);
    return true;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    // hold a strong reference to every endpoint socket for the lifetime of the
    // process: the rpc-server serves one client at a time, so reconnecting per
    // operation (what a weak_ptr cache degrades to whenever no buffer holds a
    // strong ref) floods the server with one-shot connections and starves every
    // reconnect attempt while the server is busy with a long request
    static std::unordered_map<std::string, std::shared_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        return it->second;
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
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        sock = socket_t::connect(host.c_str(), port);
        if (sock != nullptr && negotiate_hello(sock)) {
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
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
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

static rpc_stream * get_stream(const std::string & endpoint) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<rpc_stream>> streams;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = streams.find(endpoint);
    if (it != streams.end()) {
        return it->second.get();
    }
    auto s = std::make_unique<rpc_stream>(endpoint);
    rpc_stream * ptr = s.get();
    streams[endpoint] = std::move(s);
    return ptr;
}

static bool send_rpc_cmd_ordered(
        const std::string & endpoint, socket_ptr sock, enum rpc_cmd cmd,
        const void * input, size_t input_size) {
    if (!rpc_async_enabled()) {
        return send_rpc_cmd(sock, cmd, input, input_size);
    }
    return get_stream(endpoint)->call([sock, cmd, input, input_size] {
        return send_rpc_cmd(sock, cmd, input, input_size);
    });
}

static bool send_rpc_cmd_ordered(
        const std::string & endpoint, socket_ptr sock, enum rpc_cmd cmd,
        const void * input, size_t input_size, void * output, size_t output_size) {
    if (!rpc_async_enabled()) {
        return send_rpc_cmd(sock, cmd, input, input_size, output, output_size);
    }
    return get_stream(endpoint)->call([sock, cmd, input, input_size, output, output_size] {
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
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
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
    bool status = send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
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
    memset(result.padding, 0, sizeof(result.padding));

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

        bool status = send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
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
    return send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
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
    if (!send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response))) {
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

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd_ordered(ctx->endpoint, ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    // RPC has no dedicated memset command; emulate it by writing a host buffer of
    // the fill value through the normal SET_TENSOR path. Without this the buffer
    // iface's memset slot is NULL and ggml_backend_tensor_memset() aborts on any
    // tensor that lives on a board — e.g. DeepSeek-V4's DSA KV cache zeroing a
    // per-stream slice on sequence clear (llama-kv-cache-dsv4.cpp). memset happens
    // on cache clears, not per token, so the extra transfer is negligible; bypass
    // the hash cache (a plain fill is not worth a dedup round-trip).
    if (size == 0) {
        return;
    }
    std::vector<uint8_t> tmp(size, value);
    RPC_STATUS_ASSERT(rpc_buffer_set_tensor_raw(buffer, tensor, tmp.data(), offset, size));
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
            new ggml_backend_rpc_buffer_context{buft_ctx->endpoint, sock, nullptr, response.remote_ptr},
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
    if (!rpc_async_enabled()) {
        // legacy path: graph_compute is a blocking send and there are no async
        // ops in flight, so nothing to wait for
        return;
    }
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    const std::string endpoint = rpc_ctx->endpoint;
    const uint32_t device = rpc_ctx->device;
    rpc_stream * stream = get_stream(endpoint);
    // make sure the server has drained every in-order command (incl. the last
    // fire-and-forget GRAPH_COMPUTE), then wait for all worker tasks to finish.
    // when nothing was enqueued since the last completed ping the socket is
    // quiescent and the round-trip is skipped - llama_context::synchronize()
    // fires per sched x per backend and would otherwise storm the LAN with
    // redundant pings (the dominant stage-2 spec_proc overhead).
    if (stream->needs_barrier()) {
        const uint64_t covered = stream->enqueue_barrier([endpoint, device]{ rpc_ping(endpoint, device); });
        stream->drain();
        stream->mark_barrier(covered);
    } else {
        stream->drain();
    }
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
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
            tensors.push_back(serialize_tensor(t));
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

static void serialize_graph(uint32_t device, uint64_t uid, const std::string & endpoint,
                            const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
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
    std::vector<ggml_tensor *> nodes;
    nodes.reserve(cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (rpc_node_is_foreign(cgraph->nodes[i], endpoint)) {
            continue;
        }
        nodes.push_back(cgraph->nodes[i]);
    }
    uint32_t n_nodes = nodes.size();
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(nodes[i], tensors, visited);
    }
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

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    ggml_backend_dev_t rpc_dev = ggml_backend_get_device(backend);
    ggml_backend_rpc_device_context * rpc_dev_ctx = (ggml_backend_rpc_device_context *)rpc_dev->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    const std::string endpoint = rpc_ctx->endpoint;
    const uint32_t device = rpc_ctx->device;
    const bool async = rpc_async_enabled();

    // LRU over the uids the server holds deserialized for this device. A hit
    // means the server can recompute without a serialize/deserialize round.
    bool reuse = false;
    uint64_t evicted_uid = 0;
    if (cgraph->uid != 0) {
        auto & known = rpc_dev_ctx->known_graph_uids;
        auto it = std::find(known.begin(), known.end(), cgraph->uid);
        if (it != known.end()) {
            known.erase(it);
            known.insert(known.begin(), cgraph->uid);
            reuse = true;
        } else {
            known.insert(known.begin(), cgraph->uid);
            if (known.size() > RPC_GRAPH_CACHE_SLOTS) {
                evicted_uid = known.back();
                known.pop_back();
            }
        }
    }
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = device;
        request.uid    = cgraph->uid;
        if (async) {
            // enqueue on the endpoint's stream so the scheduler thread does not block
            get_stream(endpoint)->enqueue([endpoint, request]{
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
        rpc_trace_xdev(endpoint, cgraph);
        std::vector<uint8_t> input;
        serialize_graph(device, cgraph->uid, endpoint, cgraph, input);
        rpc_msg_graph_forget_req forget = { device, evicted_uid };
        if (async) {
            auto in = std::make_shared<std::vector<uint8_t>>(std::move(input));
            get_stream(endpoint)->enqueue([endpoint, in, forget]{
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
    auto msg = rpc_prepare_set_tensor(tensor, data, offset, size);
    const std::string endpoint = rpc_ctx->endpoint;
    get_stream(endpoint)->enqueue([endpoint, msg]{
        auto sock = get_socket(endpoint);
        if (sock == nullptr) {
            GGML_LOG_ERROR("[rpc set_tensor_async] lost connection to %s\n", endpoint.c_str());
            return;
        }
        send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, msg->data(), msg->size());
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
    get_stream(endpoint)->enqueue([endpoint, request, data, size]{
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
    rpc_stream * dst_stream = get_stream(dst_ctx->endpoint);
    const std::string dst_endpoint = dst_ctx->endpoint;
    // snapshot the SET message header now; the payload is filled in by the tasks below
    auto msg = rpc_prepare_set_tensor(dst, nullptr, 0, size);

    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // RPC -> RPC (star topology): read on the source stream (the server's FIFO
        // orders the read after the source's compute), hand the payload to the dst
        // stream which uploads it
        ggml_backend_rpc_context * src_ctx = (ggml_backend_rpc_context *)backend_src->context;
        rpc_stream * src_stream = get_stream(src_ctx->endpoint);
        const std::string src_endpoint = src_ctx->endpoint;
        rpc_msg_get_tensor_req get_req;
        get_req.tensor = serialize_tensor(src);
        get_req.offset = 0;
        get_req.size   = size;
        auto gate = std::make_shared<rpc_gate>();
        src_stream->enqueue([src_endpoint, get_req, msg, size, gate]{
            auto sock = get_socket(src_endpoint);
            if (sock != nullptr) {
                send_rpc_cmd(sock, RPC_CMD_GET_TENSOR, &get_req, sizeof(get_req), msg->data() + RPC_SET_TENSOR_HDR, size);
            } else {
                GGML_LOG_ERROR("[rpc cpy_tensor_async] lost connection to %s\n", src_endpoint.c_str());
            }
            gate->set();
        });
        dst_stream->enqueue([dst_endpoint, msg, gate]{
            gate->wait();
            auto sock = get_socket(dst_endpoint);
            if (sock == nullptr) {
                GGML_LOG_ERROR("[rpc cpy_tensor_async] lost connection to %s\n", dst_endpoint.c_str());
                return;
            }
            send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, msg->data(), msg->size());
        });
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
        // graph's kernels - then let the dst stream wait until the source has
        // actually produced it before shipping.
        ggml_backend_tensor_get_async(backend_src, src, msg->data() + RPC_SET_TENSOR_HDR, 0, size);
        ggml_backend_event_t ev = rpc_src_event_record(backend_src);
        dst_stream->enqueue([dst_endpoint, msg, ev, backend_src]() {
            if (ev != nullptr) {
                ggml_backend_event_synchronize(ev);
                rpc_src_event_release(ev);
            } else if (backend_src->iface.synchronize != nullptr) {
                ggml_backend_synchronize(backend_src);
            }
            auto sock = get_socket(dst_endpoint);
            if (sock == nullptr) {
                GGML_LOG_ERROR("[rpc cpy_tensor_async] lost connection to %s\n", dst_endpoint.c_str());
                return;
            }
            send_rpc_cmd(sock, RPC_CMD_SET_TENSOR, msg->data(), msg->size());
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

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir, size_t cache_limit)
        : backends(std::move(all_backends)), cache_dir(cache_dir), cache_limit(cache_limit) {
        stored_graphs.resize(backends.size());
    }
    ~rpc_server();

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool graph_forget(const rpc_msg_graph_forget_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph = nullptr;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
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
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // [fork, PipeDec] deserialized graphs kept per backend, keyed by the
    // client's graph uid. The client mirrors this set with a bounded LRU and
    // evicts explicitly via GRAPH_FORGET, so lookups on RECOMPUTE never miss.
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
};

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
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
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
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
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


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
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
    const bool cache_this = cache_pending
        && memcmp(&cache_pending_tensor, in_tensor, sizeof(rpc_tensor)) == 0
        && cache_pending_offset == offset;
    cache_pending = false;
    if (cache_dir && cache_this && size > HASH_THRESHOLD) {
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, cache_pending_hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.string().c_str());
        rpc_cache_enforce_limit(cache_dir, cache_limit);
        cache_upload_bytes += size;
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
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
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    // bump the mtime so LRU eviction sees this entry as recently used
    fs::last_write_time(cache_file, fs::file_time_type::clock::now(), ec);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    cache_pending = false;
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
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
    size_t size = cached_file.size();
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
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
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
    }
    const int64_t t_build = graph_trace ? ggml_time_us() : 0;
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    const int64_t t_compute = graph_trace ? ggml_time_us() : 0;
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
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
        fprintf(stderr,
                "[rpc graph] t=%.3f dev=%u nodes=%u tensors=%u wait=%.2fms parse=%.2fms build=%.2fms compute=%.2fms store=%.2fms total=%.2fms\n",
                epoch, device, n_nodes, n_tensors, t_wait_ms,
                (t_parse   - t_start)   / 1000.0,
                (t_build   - t_parse)   / 1000.0,
                (t_compute - t_build)   / 1000.0,
                (t_end     - t_compute) / 1000.0,
                (t_end     - t_start)   / 1000.0);
        fflush(stderr);
        t_prev_end = ggml_time_us();
    }
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
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
    ggml_status status = ggml_backend_graph_compute(backends[device], it->second.graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
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

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             size_t cache_limit, socket_ptr sock) {
    rpc_server server(backends, cache_dir, cache_limit);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
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
                response.device_count = backends.size();
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
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
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
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

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
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        char ts[32];
        {
            time_t now = time(nullptr);
            strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));
        }
        std::string peer = client_socket->peer_str();
        printf("[%s] accepted connection from %s\n", ts, peer.c_str());
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, cache_limit, client_socket);
        {
            time_t now = time(nullptr);
            strftime(ts, sizeof(ts), "%H:%M:%S", localtime(&now));
        }
        printf("[%s] connection from %s closed\n", ts, peer.c_str());
        fflush(stdout);
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
