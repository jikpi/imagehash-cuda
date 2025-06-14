#include "phash_kern.cuh"

#include <iostream>
#include <cmath>

#include <FreeImage.h>
#include "constants_common_decl.h"

// texture object for hardware-accelerated interpolation
cudaTextureObject_t createTextureObject(cudaArray_t cudaArray, int width, int height) {
    cudaResourceDesc resDesc{};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cudaArray;

    cudaTextureDesc texDesc{};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp to edge
    texDesc.addressMode[1] = cudaAddressModeClamp; // Clamp to edge
    texDesc.filterMode = cudaFilterModeLinear; // Hardware bilinear interpolation
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0; // Use pixel coordinates (non-normalized)

    cudaTextureObject_t texObj = 0;
    cudaError_t status = cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaCreateTextureObject failed for array %p (w:%d, h:%d): %s\n",
                (void *) cudaArray, width, height, cudaGetErrorString(status));
        return 0;
    }
    return texObj;
}

__global__ void preprocessImageKernel_Texture(
    cudaTextureObject_t texInput,
    float *__restrict__ outputBase,
    int inputWidth,
    int inputHeight,
    int batchOffset) {
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_out >= RESIZE_DIM || y_out >= RESIZE_DIM) {
        return;
    }

    float scale_x = (inputWidth > 1) ? (float) (inputWidth - 1) / (float) (RESIZE_DIM - 1) : 0.0f;
    float scale_y = (inputHeight > 1) ? (float) (inputHeight - 1) / (float) (RESIZE_DIM - 1) : 0.0f;

    float x_in = (float) x_out * scale_x;
    float y_in = (float) y_out * scale_y;

    // Fetch normalized float4 due to cudaReadModeNormalizedFloat
    float4 pix_norm = tex2D<float4>(texInput, x_in + 0.5f, y_in + 0.5f);

    // Convert normalized components back to approx 0-255 range.
    // FreeImage BGRA order: pix_norm.x=B, .y=G, .z=R (alpha in .w)
    // grayscale_weights: [0] = R_weight, [1] = G_weight, [2] = B_weight

    float r_component = pix_norm.z * 255.0f;
    float g_component = pix_norm.y * 255.0f;
    float b_component = pix_norm.x * 255.0f;

    float grayscale_val = grayscale_weights[0] * r_component + // R_weight * R_value
                          grayscale_weights[1] * g_component + // G_weight * G_value
                          grayscale_weights[2] * b_component; // B_weight * B_value

    int outputIndex = batchOffset + y_out * RESIZE_DIM + x_out;
    outputBase[outputIndex] = grayscale_val;
}


__global__ void dct1d_kernel(const float *__restrict__ inputBase, // Base of the batch input buffer for this stage
                             float *__restrict__ outputBase, // Base of the batch output buffer for this stage
                             int N, int M, bool row_wise,
                             int batchOffset) // Offset for "this image" within inputBase and outputBase
{
    // Global indices for the element being computed within the logical N x M matrix of *this image*
    int k = blockIdx.x * blockDim.x + threadIdx.x; // Index along the dimension being transformed (0..N-1 or 0..M-1)
    int vec_idx = blockIdx.y; // Index of the row/column being processed within this image (0..M-1 or 0..N-1)

    // find the actual dimensions based on row/column transform
    int transformLength = row_wise ? N : M; // Length of the 1D vector being transformed
    int numVectors = row_wise ? M : N; // Number of 1D vectors to process
    int outputDim1 = row_wise ? M : N; // First dimension of output array (num vectors)
    int outputDim2 = row_wise ? N : M; // Second dimension of output array (transformed vector length)

    // Check bounds: k is the index within the transformed vector (outputDim2), vec_idx is the vector index (outputDim1)
    if (k >= outputDim2 || vec_idx >= outputDim1) {
        return;
    }

    float sum = 0.0f;
    // Scaling factor for DCT basis k. 'transformLength' is the correct dimension here.
    float scale_k = (k == 0) ? sqrtf(1.0f / transformLength) : sqrtf(2.0f / transformLength);

    // Loop over the elements (n) of the input vector being transformed
    for (int n = 0; n < transformLength; ++n) {
        float input_val;
        int inputIndex; // Index relative to the start of *this image's data* in the batch buffer
        if (row_wise) {
            // Processing row vec_idx: read input[vec_idx * N + n]
            inputIndex = vec_idx * N + n;
        } else {
            // Processing column vec_idx: read input[n * M + vec_idx]
            // Input here is the output of the row-wise step. M=N=RESIZE_DIM.
            // Layout is row-major (M rows, N cols). Access column vec_idx at row n.
            inputIndex = n * N + vec_idx; // Correct for row-major layout if M=N
        }
        // Add the batchOffset to get the absolute index in the global batch buffer
        input_val = inputBase[batchOffset + inputIndex];

        // Use pre-computed cosine value from constant memory
        // dct_cos_table[k * RESIZE_DIM + n]
        float cos_val = dct_cos_table[k * RESIZE_DIM + n];
        sum += input_val * cos_val;
    }

    float result = scale_k * sum;

    // Calculate output index relative to the start of "this image's data"
    int outputIndexRelative;
    if (row_wise) {
        // Writing row vec_idx: output[vec_idx * N + k]
        outputIndexRelative = vec_idx * N + k;
    } else {
        // Writing column vec_idx: output[k * M + vec_idx] (Transposed write)
        // Output logically N x M. Write to row k, column vec_idx.
        outputIndexRelative = k * M + vec_idx; // Correct for row-major layout if M=N
    }
    // Add the batchOffset to get the absolute index in the global batch buffer
    outputBase[batchOffset + outputIndexRelative] = result;
}


__global__ void computeHashBitsKernel(const float *__restrict__ dctCoeffsBase, // Base of the entire batch of DCT coeffs
                                      uint64_t *__restrict__ finalHashBase, // Base of the entire batch of hash outputs
                                      int validImagesInBatch) // Number of valid images this grid is processing
{
    // Use blockIdx.x to determine which image in the "valid subset" this block processes
    int imageIndexInBatch = blockIdx.x;

    // Check if this block corresponds to a valid image index for this launch
    if (imageIndexInBatch >= validImagesInBatch) {
        return;
    }

    // Shared memory for DCT_REDUCE_DIM x DCT_REDUCE_DIM coefficients and reduction sum
    // One block processes one image.
    extern __shared__ float sharedMem[]; // Declare dynamic shared memory
    float *sharedCoeffs = sharedMem; // First part for coeffs (64 floats)
    float *sharedSum = &sharedMem[HASH_BITS]; // Second part for sum (1 float)

    int tx = threadIdx.x; // 0..DCT_REDUCE_DIM-1 (0..7)
    int ty = threadIdx.y; // 0..DCT_REDUCE_DIM-1 (0..7)
    // Local thread ID within the 8x8 block (0..63)
    int threadId = ty * blockDim.x + tx;

    // Calculate the base offset for "this image's" DCT coefficients in the global batch array
    // Input dctCoeffsBase is N x M (RESIZE_DIM x RESIZE_DIM) per image
    const float *imageDctCoeffs = dctCoeffsBase + (ptrdiff_t) imageIndexInBatch * RESIZE_DIM * RESIZE_DIM;

    // 1. Load top-left DCT_REDUCE_DIM x DCT_REDUCE_DIM coefficients into shared memory ---
    // read from the correct image's data in the batch
    // indexing into the full RESIZE_DIM x RESIZE_DIM matrix for this image
    sharedCoeffs[threadId] = imageDctCoeffs[ty * RESIZE_DIM + tx];

    __syncthreads();

    // 2. Calculate the average of the DCT_REDUCE_DIM * DCT_REDUCE_DIM coefficients (excluding DC term at [0,0]) ---
    // Parallel reduction in shared memory.

    // Each thread initializes its contribution to the sum (DC term contributes 0)
    float myVal = (threadId == 0) ? 0.0f : sharedCoeffs[threadId];
    // Store it back into shared memory for reduction
    sharedCoeffs[threadId] = myVal;
    __syncthreads();

    // Perform reduction in shared memory (logarithmic steps)
    for (int offset = blockDim.x * blockDim.y / 2; offset > 0; offset >>= 1) {
        if (threadId < offset) {
            // Add value from threadId + offset to threadId
            sharedCoeffs[threadId] += sharedCoeffs[threadId + offset];
        }
        __syncthreads();
    }

    // After loop, thread 0 holds the total sum in sharedCoeffs[0]
    // Thread 0 writes the sum to the designated sharedSum[0] location (optional, could just use sharedCoeffs[0])
    if (threadId == 0) {
        sharedSum[0] = sharedCoeffs[0];
    }
    __syncthreads();

    // Calculate Average
    // All threads read the final sum calculated by thread 0
    float average = 0.0f;
    if (HASH_BITS > 1) {
        average = sharedSum[0] / (float) (HASH_BITS - 1);
    }

    // 3. Compute the bit for this thread's coefficient ---
    uint64_t myBit = 0;
    // Compare this thread's original coefficient (re-read from shared memory) against the average
    if (sharedCoeffs[threadId] > average) {
        // Use the coefficient value loaded at the start
        // Set the corresponding bit. Bit order convention:
        // (0,0) -> bit 63, (0,1) -> bit 62, ..., (7,7) -> bit 0
        // This maps threadId 0..63 to bit position 63..0
        myBit = 1ULL << (HASH_BITS - 1 - threadId);
    }

    // 4. Combine bits into the final hash for this image using atomicOr ---
    // Each thread computes its bit and atomically ORs it into the
    // correct slot in the global finalHashBase array corresponding to this image (imageIndexInBatch).
    atomicOr((unsigned long long *) &finalHashBase[imageIndexInBatch], (unsigned long long) myBit);
}

cudaError_t initializeConstantMemory() {
    cudaError_t status = cudaSuccess;

    // grayscale weights: R=0.299, G=0.587, B=0.114
    float h_grayscale_weights[3] = {0.299f, 0.587f, 0.114f};
    status = cudaMemcpyToSymbol(grayscale_weights, h_grayscale_weights, 3 * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy grayscale weights to constant memory: " << cudaGetErrorString(status) << std::endl;
        return status;
    }

    // Pre-compute DCT cosine table
    float *h_dct_cos_table = new float[RESIZE_DIM * RESIZE_DIM];

    // Fill the cosine table for all possible (k, n) combinations
    for (int k = 0; k < RESIZE_DIM; ++k) {
        for (int n = 0; n < RESIZE_DIM; ++n) {
            h_dct_cos_table[k * RESIZE_DIM + n] = cosf(PI * (n + 0.5f) * k / (float) RESIZE_DIM);
        }
    }

    // Copy to constant memory
    status = cudaMemcpyToSymbol(dct_cos_table, h_dct_cos_table, RESIZE_DIM * RESIZE_DIM * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "Failed to copy DCT cosine table to constant memory: " << cudaGetErrorString(status) << std::endl;
        delete[] h_dct_cos_table;
        return status;
    }

    delete[] h_dct_cos_table;
    return cudaSuccess;
}
