#pragma once

#include "llama.h"

#include "llama-impl.h"
#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-mmap.h"

#include "ggml-cpp.h"

#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using llama_buf_map = std::unordered_map<uint32_t, ggml_backend_buffer_t>;
using llama_ctx_buf_maps = std::vector<std::pair<ggml_context *, llama_buf_map>>;

// lists of buffer types used for each layer
using buft_list_t = std::vector<std::pair<ggml_backend_dev_t, ggml_backend_buffer_type_t>>;

enum llama_fver {
    GGUF_FILE_VERSION_V1 = 1,
    GGUF_FILE_VERSION_V2 = 2,
    GGUF_FILE_VERSION_V3 = 3,
};

const char * llama_file_version_name(llama_fver version);

// ---------------------------------------------------------------------------
// RPC weight loading  [fork]
//
// Shared by the initial model load and by the fleet-hibernation reload, which
// has no llama_model_loader to work from: it hands over the same three things
// a load does - the weight contexts, the model file paths, and a way to look
// up where a tensor's payload sits in them.
// ---------------------------------------------------------------------------

// entry points on the RPC backend, resolved by proc address so this code does
// not depend on the RPC backend being present
struct llama_rpc_cache_api {
    using query_fn_t    = int  (*)(ggml_backend_buffer_t, ggml_tensor *, size_t, size_t, uint64_t);
    using upload_fn_t   = bool (*)(ggml_backend_buffer_t, ggml_tensor *, const void *, size_t, size_t);
    using endpoint_fn_t = const char * (*)(ggml_backend_buffer_t);

    query_fn_t    query    = nullptr;
    upload_fn_t   upload   = nullptr;
    endpoint_fn_t endpoint = nullptr;
    size_t        threshold = 0;
};

// the weight contexts that live on one RPC endpoint
struct llama_rpc_job {
    ggml_backend_dev_t device = nullptr;
    std::string endpoint;
    llama_rpc_cache_api api;
    std::vector<ggml_context *> contexts;
};

// where a weight tensor's payload sits in the model files
struct llama_rpc_weight_source {
    uint16_t idx  = 0;   // file index
    size_t   offs = 0;
};

using llama_rpc_source_fn = std::function<bool(const char * name, llama_rpc_weight_source & out)>;

std::vector<llama_rpc_job> llama_rpc_jobs_for(
        const std::vector<ggml_context *> & contexts,
        std::unordered_set<ggml_context *> * matched);

bool llama_rpc_upload_weights(
        const std::vector<llama_rpc_job> & jobs,
        const std::vector<std::string> & file_paths,
        const llama_rpc_source_fn & source_of,
        bool check_tensors,
        const std::function<bool()> & concurrent_work,
        std::unordered_set<const ggml_tensor *> * preloaded,
        std::string & err);

struct llama_model_loader {
    // Holds information on a model weight
    struct llama_tensor_weight {
        uint16_t  idx; // source file index
        size_t   offs; // tensor data offset in the original file

        ggml_tensor * tensor;

        llama_tensor_weight(const llama_file * file, uint16_t idx, const struct gguf_context * gguf_ctx, ggml_tensor * tensor) : idx(idx), tensor(tensor) {
            const int tensor_idx = gguf_find_tensor(gguf_ctx,  ggml_get_name(tensor));
            if (tensor_idx < 0) {
                throw std::runtime_error(format("tensor '%s' not found in the model", ggml_get_name(tensor)));
            }

            offs = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tensor_idx);
            if (offs + ggml_nbytes(tensor) < offs || offs + ggml_nbytes(tensor) > file->size()) {
                throw std::runtime_error(format("tensor '%s' data is not within the file bounds, model is corrupted or incomplete", ggml_get_name(tensor)));
            }
        }
    };

    // custom comparator to sort weights more nicely by layer
    struct weight_name_comparer {
        bool operator()(const std::string & a, const std::string & b) const {
            int a_layer = -1;
            int b_layer = -1;
            sscanf(a.c_str(), "blk.%d.", &a_layer);
            sscanf(b.c_str(), "blk.%d.", &b_layer);
            if (a_layer != b_layer) {
                return a_layer < b_layer;
            }
            return a < b;
        }
    };

    static const int TENSOR_NOT_REQUIRED    = 1 << 0;
    static const int TENSOR_DUPLICATED      = 1 << 1;
    static const int TENSOR_SKIP            = 1 << 2;
    static const int TENSOR_SKIP_IF_VIRTUAL = 1 << 3;
    static const int TENSOR_ALLOW_RESHAPE   = 1 << 4;
    static const int TENSOR_READ_LAZY       = 1 << 5; // read rows on demand instead of loading whole tensor; requires mmap for now

    int n_kv      = 0;
    int n_tensors = 0;
    int n_created = 0;

    uint64_t n_elements = 0;
    size_t   n_bytes    = 0;

    bool use_mmap = false;
    bool use_direct_io = false;
    bool check_tensors;
    bool no_alloc;
    bool load_mtp;

    // set by the caller before the create_tensor() calls
    enum llama_lazy_mode lazy_mode = LLAMA_LAZY_MODE_OFF;

    llama_files files;
    std::vector<std::string> file_paths;
    llama_ftype ftype;
    llama_fver  fver;

    llama_mmaps mappings;

    // byte ranges of TENSOR_READ_LAZY tensors, per file index
    std::map<uint32_t, llama_mmap::ranges> lazy_tensor_ranges;

    std::map<std::string, llama_tensor_weight, weight_name_comparer> weights_map;
    std::unordered_map<std::string, llama_model_kv_override> kv_overrides;
    const llama_model_tensor_buft_override * tensor_buft_overrides;

    gguf_context_ptr metadata_ptr;
    struct gguf_context * metadata; // either metadata_ptr.get() or externally set
    llama_model_set_tensor_data_t set_tensor_data;
    void * set_tensor_data_ud;
    std::vector<ggml_context_ptr> contexts;

    std::string arch_name;
    LLM_KV      llm_kv    = LLM_KV(LLM_ARCH_UNKNOWN);

    size_t size_done = 0;
    size_t size_data = 0;
    std::vector<std::pair<size_t, size_t>> mmaps_used;
    std::unordered_set<const ggml_tensor *> rpc_preloaded;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };

    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // track tensors that had to be moved for debugging:
    size_t n_tensors_moved = 0;
    std::string first_tensor_moved_name;
    std::string first_tensor_moved_type_name;
    ggml_backend_buffer_type_t first_moved_from_buft = nullptr;
    ggml_backend_buffer_type_t first_moved_to_buft = nullptr;

    llama_model_loader(
        struct gguf_context * metadata,
        llama_model_set_tensor_data_t set_tensor_data,
        void * set_tensor_data_ud,
        const std::string & fname,
        std::vector<std::string> & splits, // optional, only need if the split does not follow naming scheme
        FILE * file,
        llama_load_mode load_mode,
        bool check_tensors,
        bool no_alloc,
        bool load_mtp,
        const llama_model_kv_override * param_overrides_p,
        const llama_model_tensor_buft_override * param_tensor_buft_overrides_p);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(const std::string & key, T & result, bool required = true);

    template<typename T>
    typename std::enable_if<std::is_integral<T>::value, bool>::type
    get_arr_n(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_arr(const std::string & key, std::vector<T> & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_arr(const std::string & key, std::array<T, N_MAX> & result, bool required = true);

    template<typename T>
    bool get_arr(enum llm_kv kid, T & result, bool required = true);

    template<typename T>
    bool get_key(const std::string & key, T & result, bool required = true);

    template<typename T>
    bool get_key(enum llm_kv kid, T & result, bool required = true);

    template<typename T, size_t N_MAX>
    bool get_key_or_arr(const std::string & key, std::array<T, N_MAX> & result, uint32_t n, bool required = true);

    template<typename T>
    bool get_key_or_arr(enum llm_kv kid, T & result, uint32_t n, bool required = true);

    bool get_key_or_arr(enum llm_kv kid, uint32_t & result, bool required = true);

    std::string get_arch_name() const;

    enum llm_arch get_arch() const;

    const llama_tensor_weight * get_weight(const char * name) const;

    const llama_tensor_weight & require_weight(const char * name) const;

    struct ggml_tensor * get_tensor_meta(const char * name) const;

    struct ggml_tensor * require_tensor_meta(const std::string & name) const;

    const struct ggml_tensor * check_tensor_dims(
            const std::string & name,
            const std::vector<int64_t> & ne,
            bool required,
            bool allow_reshape) const;

    struct ggml_tensor * create_tensor(
        const llama_hparams & hparams, const buft_list_t * buft_list_cpu, const buft_list_t * buft_list_input, const buft_list_t * buft_list_output,
        const buft_list_t * buft_list_layer, const LLM_TN_IMPL & tn, const std::initializer_list<int64_t> & ne, int flags);

    void done_getting_tensors(bool partial = false) const;

    void init_mappings(bool prefetch = true, llama_mlocks * mlock_mmaps = nullptr);

    void get_mapping_range(size_t * first, size_t * last, void ** addr, int idx, ggml_context * ctx) const;

    // release a weight's mmap pages
    void unmap_weight(const llama_tensor_weight & w) const;

    // read a byte range of a weight's data
    // with mmap, returns a pointer into the mapping, otherwise reads into buf and returns buf
    const void * load_data_range(const llama_tensor_weight & w, size_t offs, size_t size, void * buf) const;

    // Returns false if cancelled by progress_callback
    bool load_all_data(
            struct ggml_context * ctx,
            llama_buf_map & bufs,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    // Loads independent RPC devices concurrently. Cache-manifest hits are
    // resolved before reading GGUF payloads; non-RPC contexts continue loading
    // on the calling thread while the remote workers run.
    bool load_all_data_parallel(
            llama_ctx_buf_maps & ctx_buf_maps,
            llama_mlocks * lmlocks,
            llama_progress_callback progress_callback,
            void * progress_callback_user_data);

    std::string ftype_name() const;

    void print_info() const;
};
