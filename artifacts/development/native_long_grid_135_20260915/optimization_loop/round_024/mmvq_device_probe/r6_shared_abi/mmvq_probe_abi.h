#pragma once
#include <cuda_runtime_api.h>
// Shared by the generated definitions and every driver; no independent prototype.
extern "C" void heterollm_mmvq_probe_convert_q8_1(
    const float * src, void * q8_1, int ggml_type_id, int k, int m,
    int padded_k, cudaStream_t stream);
extern "C" void heterollm_mmvq_probe_main(
    const void * packed_weight, int ggml_type_id, const void * q8_1,
    float * output, int k, int n, int m, int q8_1_stride_blocks,
    cudaStream_t stream);
