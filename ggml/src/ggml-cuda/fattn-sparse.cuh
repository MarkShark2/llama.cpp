#pragma once

#include "common.cuh"

// [fork] DSA sparse flash attention: flash_attn_ext extended with a gathered
// top-k KV segment (src[5]=kc, src[6]=idx I32, src[7]=nvis I32) sharing one
// online softmax with the regular dense (masked) K/V. See ggml.c
// ggml_flash_attn_ext_set_sparse for the contract.

bool ggml_cuda_fattn_sparse_is_sparse_node(const ggml_tensor * dst);
bool ggml_cuda_fattn_sparse_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
