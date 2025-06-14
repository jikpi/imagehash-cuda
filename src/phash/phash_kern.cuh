#ifndef PHASH_CUH
#define PHASH_CUH

#include <cuda_runtime.h>

#include "constants_common_decl.h"

struct FIBITMAP;

#include "../helpers/FreeImageBitmap.h"

// Constant memory
__constant__ inline float dct_cos_table[RESIZE_DIM * RESIZE_DIM];
__constant__ inline float grayscale_weights[3]; // {0.299f, 0.587f, 0.114f}

// Constant memory init function
cudaError_t initializeConstantMemory();

// Texture object creation helper
cudaTextureObject_t createTextureObject(cudaArray_t cudaArray, int width, int height);

// Kernel Declarations

// Preprocessing: BGRA to Grayscale + Hardware-accelerated Resize using Texture
__global__ void preprocessImageKernel_Texture(
    cudaTextureObject_t texInput, // Texture object for hardware interpolation
    float *__restrict__ outputBase, // Pointer to the start of d_resizedGrayscaleBatch
    int inputWidth,
    int inputHeight,
    int batchOffset // Offset into outputBase (batchIndex * RESIZE_DIM * RESIZE_DIM)
);

// 1D DCT Kernel
__global__ void dct1d_kernel(
    const float *__restrict__ inputBase, // Start of d_resizedGrayscaleBatch or d_dctIntermediateBatch
    float *__restrict__ outputBase, // Start of d_dctIntermediateBatch or d_dctResultBatch
    int N, // RESIZE_DIM
    int M, // RESIZE_DIM
    bool row_wise,
    int batchOffset // Offset into input/output bases (batchIndex * N * M)
);

// Hash Computation Kernel
__global__ void computeHashBitsKernel(
    const float *__restrict__ dctCoeffsBase, // Start of d_dctResultBatch
    uint64_t *__restrict__ finalHashBase, // Start of d_hashResultBatch
    int batchSize // Number of valid images in this specific batch launch
);


#endif // PHASH_CUH
