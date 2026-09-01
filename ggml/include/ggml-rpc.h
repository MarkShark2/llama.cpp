#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    5
#define RPC_PROTO_MINOR_VERSION    1
// patch 1: full-duplex transfer lanes (SESSION_INFO / LANE_ATTACH / LANE_FENCE).
// patch 2: direct remote->remote transfer (PEER_OPEN / PUSH_TENSOR).
// patch 3: server-side imatrix reduction (IMATRIX_SQSUM).
// patch 4: parked sessions (SESSION_DETACH / SESSION_RESUME) for fleet
//          hibernation - the client disconnects, the server keeps the
//          buffers, and the KV cache survives the host suspending to disk.
// The patch level is not checked by the HELLO handshake, so old clients keep
// working against new servers; new clients only attach lanes when the server
// reports patch >= 1 (see GGML_RPC_FDX_MIN_PATCH), only route peer traffic
// when both endpoints report patch >= 2 (GGML_RPC_PEER_MIN_PATCH), and only
// reduce imatrix activations remotely at patch >= 3 (GGML_RPC_IMAT_MIN_PATCH),
// and only park a session at patch >= 4 (GGML_RPC_HIBERNATE_MIN_PATCH).
#define RPC_PROTO_PATCH_VERSION    4

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

// opt this client in to server-side tensor caching: only then are large tensors
// hashed and offered to the server's local cache (server must run with -c)
GGML_BACKEND_API void ggml_backend_rpc_set_client_cache(bool enabled);

// Helpers for model loaders that can identify a cached tensor before reading
// its payload. cache_query returns 1 for a hit, 0 for a miss, and -1 for a
// transport error. After a miss, cache_upload sends the payload without a
// second hash or query.
GGML_BACKEND_API bool   ggml_backend_rpc_get_client_cache(void);
GGML_BACKEND_API size_t ggml_backend_rpc_cache_threshold(void);
GGML_BACKEND_API int    ggml_backend_rpc_buffer_cache_query(
        ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
        size_t offset, size_t size, uint64_t hash);
GGML_BACKEND_API bool   ggml_backend_rpc_buffer_cache_upload(
        ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
        const void * data, size_t offset, size_t size);

// cache_limit caps the total size in bytes of the cache_dir contents; least
// recently used entries are evicted when the cap is exceeded (0 = unlimited)
GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir, size_t cache_limit,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// Fleet hibernation. detach parks every connected session on its server and
// closes the connections, so the remote hosts can suspend to disk without
// losing their buffers; reattach dials them again and re-adopts those
// sessions, polling until they all answer or timeout_ms passes. detach
// returns the number parked (-1 if any server refused); reattach returns the
// number resumed, or a negative count of unresolved endpoints. session_lost
// distinguishes an unreachable endpoint from an unrecoverable parked session.
// The caller must have quiesced all RPC traffic first.
GGML_BACKEND_API int  ggml_backend_rpc_detach(void);
GGML_BACKEND_API int  ggml_backend_rpc_reattach(int timeout_ms);
GGML_BACKEND_API bool ggml_backend_rpc_is_detached(void);
GGML_BACKEND_API bool ggml_backend_rpc_session_lost(void);
// fills out_names/out_connected with up to max entries, returns the total
GGML_BACKEND_API int  ggml_backend_rpc_endpoint_status(const char ** out_names, int * out_connected, int max);

#ifdef  __cplusplus
}
#endif
