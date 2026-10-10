#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    // half-close both directions so a thread blocked in recv_data on this
    // socket wakes up with an error (the fd itself is closed by the dtor)
    void shutdown_rw();

    socket_ptr accept();

    // "ip:port" of the remote end, for logging
    std::string peer_str() const;
    std::string local_str() const;

    struct command_trace_context {
        uint64_t run = 0, wave = 0, dispatch = 0;
        int stage = -1, pos = -1, active = 0;
        double lock_ms = 0, queue_ms = 0;
    };
    struct command_trace_state {
        uint64_t ordinal = 0, eligible = 0;
        bool armed = false;
        bool captured = false, last = false, response_pending = false;
        int command = 0;
        command_trace_context tag;
        std::string local, peer;
    };
    // Each direction has one owner, like the transport's send and receive buffers.
    command_trace_state trace_tx, trace_rx;

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
