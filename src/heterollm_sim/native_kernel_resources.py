"""Exact native MMVQ observations; no shape extrapolation or timing calibration.

Evidence: artifacts/development/native_kernel_binding_20260929/occupancy.json.
cuFuncGetAttribute + cuOccupancyMaxActiveBlocksPerMultiprocessor on extracted
native sm_120a cubin; CUPTI launch geometry cross-checked. Hardware RTX5080.
"""
BINARY_SHA256 = '8a7275a273c225639a94c6cd544d3a891760f4599bfbe5f6ab1b4d15f556a297'
OBSERVED = {('q4_k', 1, 4096, 4096): (53, 384, 0, 1024, 128, 9, '_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('q4_k', 2, 4096, 4096): (80, 1536, 0, 1024, 128, 6, '_Z13mul_mat_vec_qIL9ggml_type12ELi2ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('q4_k', 4, 4096, 4096): (120, 3072, 0, 1024, 128, 4, '_Z13mul_mat_vec_qIL9ggml_type12ELi4ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('q4_k', 1, 3072, 3072): (53, 384, 0, 1024, 128, 9, '_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('q6_k', 1, 4096, 4096): (56, 384, 0, 1024, 128, 9, '_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('q6_k', 2, 4096, 4096): (64, 1536, 0, 1024, 128, 8, '_Z13mul_mat_vec_qIL9ggml_type14ELi2ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('q6_k', 4, 4096, 4096): (80, 3072, 0, 1024, 128, 6, '_Z13mul_mat_vec_qIL9ggml_type14ELi4ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('q6_k', 1, 3072, 3072): (56, 384, 0, 1024, 128, 9, '_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('iq4_xs', 1, 4096, 4096): (47, 384, 0, 1024, 128, 10, '_Z13mul_mat_vec_qIL9ggml_type23ELi1ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('iq4_xs', 2, 4096, 4096): (79, 1536, 0, 1024, 128, 6, '_Z13mul_mat_vec_qIL9ggml_type23ELi2ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('iq4_xs', 4, 4096, 4096): (104, 3072, 0, 1024, 128, 4, '_Z13mul_mat_vec_qIL9ggml_type23ELi4ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('iq4_xs', 8, 4096, 4096): (146, 2048, 0, 1024, 64, 6, '_Z13mul_mat_vec_qIL9ggml_type23ELi8ELb0ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj'), ('iq4_xs', 1, 3072, 3072): (79, 1536, 0, 1024, 128, 6, '_Z13mul_mat_vec_qIL9ggml_type23ELi1ELb0ELb1ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj')}


def observed_mmvq_resources(source, *, hardware_id):
    if hardware_id != 'nvidia-rtx-5080' or source.runtime_binary_sha256 != BINARY_SHA256:
        return None
    # Compiled resource allocation depends on the full specialization and block,
    # not N or the number of iterations of its runtime K loop.
    from .mmvq_work import MMVQSourceContract, SOURCE_SHA256, derive_mmvq_work, UnsupportedMMVQ
    try:
        expected = derive_mmvq_work(m=source.m,n=source.n,k=source.k,weight_format=source.weight_format,
            contract=MMVQSourceContract(1200,1200,32,SOURCE_SHA256,BINARY_SHA256,True,False,True),
            allow_k_formats=True, allow_iq4_xs=True)
    except UnsupportedMMVQ:
        return None
    if source != expected:
        return None
    # If the exact format/M/block specialization is present in the binary artifact
    # but not in the first Nsight sample matrix, refuse to infer it here.
    if source.weight_format.lower() == 'q5_k' and source.m == 1:
        return None
    candidates = {row for key, row in OBSERVED.items()
                  if key[:2] == (source.weight_format.lower(), source.m)
                  and f'ELb0ELb{int(source.small_k)}ELb0E' in row[-1]
                  and row[4] == source.warps_per_cta * 32}
    if len(candidates) != 1:
        return None
    row = candidates.pop()
    registers, static, dynamic, reserved, threads, resident, symbol = row
    # Full ordinary/unfused specialization is constrained by MMVQSourceContract.
    # Check the small-K flag and actual launch, not merely M.
    if f'ELb0ELb{int(source.small_k)}ELb0E' not in symbol:
        return None
    if source.grid != ((source.n + source.rows_per_cta - 1) // source.rows_per_cta, 1, 1):
        return None
    if source.block != (32, threads // 32, 1) or source.warps_per_cta * 32 != threads:
        return None
    return dict(registers=registers, static_shared=static, dynamic_shared=dynamic,
                reserved_shared=reserved, resident_ctas=resident, symbol=symbol,
                binary_sha256=BINARY_SHA256, scope='exact_binary_specialization_and_block_runtime_dimensions')
