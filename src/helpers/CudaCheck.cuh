//
// Created by skl on 2025-06-13.
//

#ifndef CUDACHECK_CUH
#define CUDACHECK_CUH

// --- CUDA Check Macro ---
#define CUDA_CHECK(call) do { \
cudaError_t macro_err_ = call; \
if (macro_err_ != cudaSuccess) { \
fprintf(stderr, "CUDA Error at %s:%d (in function %s) - %s (%d)\n", __FILE__, __LINE__, __func__, cudaGetErrorString(macro_err_), macro_err_); \
status = macro_err_; \
goto compute_hashes_cleanup; \
} \
} while (0)

#define CUDA_CHECK_KERNEL() do { \
cudaError_t macro_err_ = cudaGetLastError(); \
if (macro_err_ != cudaSuccess) { \
fprintf(stderr, "CUDA Kernel Error at %s:%d (in function %s) - %s (%d)\n", __FILE__, __LINE__, __func__, cudaGetErrorString(macro_err_), macro_err_); \
status = macro_err_; \
goto compute_hashes_cleanup; \
} \
} while (0)

#endif //CUDACHECK_CUH
