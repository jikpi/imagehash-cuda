//
// Created by skl on 2025-06-13.
//

#include "phash.cuh"
#include <cuda_runtime.h>

#include <FreeImage.h>
#include <algorithm>
#include <iostream>

#include "constants_common_decl.h"
#include "constants_host_decl.h"
#include "../helpers/CudaCheck.cuh"
#include "phash_kern.cuh"

HostBatchBuffer::HostBatchBuffer() : imageBitmaps(MAX_BATCH_INSTANCES),
                                     imagePaths(MAX_BATCH_INSTANCES),
                                     imageMetadata(MAX_BATCH_INSTANCES),
                                     imageValid(MAX_BATCH_INSTANCES, false) {
}

void HostBatchBuffer::clear() {
    if (batchSize > 0) {
        for (size_t i = 0; i < batchSize; ++i) {
            imageBitmaps[i].reset();
            imagePaths[i].clear();
            imageMetadata[i] = {};
            imageValid[i] = false;
        }
        batchSize = 0;
    }
}

// computeHashes function implementation
ComputeHashesResult computeHashes(
    const std::vector<std::filesystem::path> &allImagePaths,
    std::atomic<size_t> &failedImageLoads,
    std::atomic<size_t> &failedKernelExecutions
) {
    auto processingStartTime = std::chrono::high_resolution_clock::now();
    cudaError_t status = cudaSuccess;

    // Initialize constant memory
    status = initializeConstantMemory();
    if (status != cudaSuccess) {
        std::cerr << "Failed to initialize constant memory: " << cudaGetErrorString(status) << std::endl;
        ComputeHashesResult result;
        return result; // Return early on failure
    }

    ComputeHashesResult result;
    result.imagesSubmittedForProcessing = 0;

    size_t totalImages = allImagePaths.size();
    size_t totalBatches = (totalImages + MAX_BATCH_INSTANCES - 1) / MAX_BATCH_INSTANCES;
    result.hashes.reserve(totalImages);

    HostBatchBuffer hostBuffer; // Instance of the batch buffer

    // Device memory pointers
    std::vector<cudaArray_t> d_inputArrays(MAX_BATCH_INSTANCES, nullptr);
    std::vector<cudaTextureObject_t> d_textureObjects(MAX_BATCH_INSTANCES, 0);
    float *d_resizedGrayscaleBatch = nullptr;
    float *d_dctIntermediateBatch = nullptr;
    float *d_dctResultBatch = nullptr;
    uint64_t *d_hashResultBatch = nullptr;
    uint64_t *h_pinnedHashOutput = nullptr; // Single pinned host buffer for results

    // Batch size variables for GPU memory
    size_t grayBatchSize = (size_t) MAX_BATCH_INSTANCES * RESIZE_DIM * RESIZE_DIM * sizeof(float);
    size_t dctBatchSize = grayBatchSize;
    size_t hashBatchItemSize = sizeof(uint64_t);
    size_t hashBatchTotalSize = (size_t) MAX_BATCH_INSTANCES * hashBatchItemSize;

    // Shared memory size for hash kernel
    size_t sharedMemSize = (HASH_BITS + 1) * sizeof(float); // 64 coeffs + 1 sum = 65 floats

    // Channel descriptor for uchar4 (BGRA)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();

    // Device-Side Memory Allocation ---
    status = cudaMalloc(&d_resizedGrayscaleBatch, grayBatchSize);
    if (status != cudaSuccess) {
        std::cerr << "Failed cudaMalloc d_resizedGrayscaleBatch\n";
        goto compute_hashes_cleanup;
    }

    status = cudaMalloc(&d_dctIntermediateBatch, dctBatchSize);
    if (status != cudaSuccess) {
        std::cerr << "Failed cudaMalloc d_dctIntermediateBatch\n";
        goto compute_hashes_cleanup;
    }

    status = cudaMalloc(&d_dctResultBatch, dctBatchSize);
    if (status != cudaSuccess) {
        std::cerr << "Failed cudaMalloc d_dctResultBatch\n";
        goto compute_hashes_cleanup;
    }

    status = cudaMalloc(&d_hashResultBatch, hashBatchTotalSize);
    if (status != cudaSuccess) {
        std::cerr << "Failed cudaMalloc d_hashResultBatch\n";
        goto compute_hashes_cleanup;
    }

    status = cudaHostAlloc(&h_pinnedHashOutput, hashBatchTotalSize, cudaHostAllocDefault);
    if (status != cudaSuccess) {
        std::cerr << "Failed cudaHostAlloc h_pinnedHashOutput\n";
        goto compute_hashes_cleanup;
    }

    // Initialize CUDA arrays and texture objects
    for (size_t i = 0; i < MAX_BATCH_INSTANCES; ++i) {
        d_inputArrays[i] = nullptr;
        d_textureObjects[i] = 0;
    }

    std::cout << "Starting processing..." << std::endl;

    for (size_t batchIndex = 0; batchIndex < totalBatches; ++batchIndex) {
        size_t currentBatchStartIndex = batchIndex * MAX_BATCH_INSTANCES;
        size_t currentBatchActualSize = std::min((size_t) MAX_BATCH_INSTANCES, totalImages - currentBatchStartIndex);

        std::cout << "Processing Batch " << batchIndex + 1 << "/" << totalBatches
                << " (Images " << currentBatchStartIndex << "-" << currentBatchStartIndex + currentBatchActualSize - 1
                << ")" << std::endl;

        hostBuffer.clear();
        hostBuffer.batchSize = currentBatchActualSize;
        size_t validImagesInBatchHost = 0;

        // Load images into host buffer
        for (size_t i = 0; i < currentBatchActualSize; ++i) {
            const auto &imgPath = allImagePaths[currentBatchStartIndex + i];
            hostBuffer.imagePaths[i] = imgPath;
            FIBITMAP *tempBmp = nullptr;
            FIBITMAP *tempBmp32 = nullptr;
            std::string filenameStr = imgPath.string();
            const char *filename = filenameStr.c_str();

            try {
                FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename, 0);
                if (format == FIF_UNKNOWN) format = FreeImage_GetFIFFromFilename(filename);

                if (format != FIF_UNKNOWN && FreeImage_FIFSupportsReading(format)) {
                    tempBmp = FreeImage_Load(format, filename);
                    if (tempBmp) {
                        tempBmp32 = FreeImage_ConvertTo32Bits(tempBmp);
                        FreeImage_Unload(tempBmp);
                        tempBmp = nullptr;
                        if (tempBmp32) {
                            hostBuffer.imageBitmaps[i].reset(tempBmp32);
                            hostBuffer.imageMetadata[i] = {
                                FreeImage_GetWidth(tempBmp32), FreeImage_GetHeight(tempBmp32),
                                FreeImage_GetPitch(tempBmp32),
                                (size_t) FreeImage_GetPitch(tempBmp32) * FreeImage_GetHeight(tempBmp32)
                            };
                            hostBuffer.imageValid[i] = true;
                            validImagesInBatchHost++;
                        } else {
                            failedImageLoads++;
                            hostBuffer.imageValid[i] = false;
                        }
                    } else {
                        failedImageLoads++;
                        hostBuffer.imageValid[i] = false;
                    }
                } else {
                    failedImageLoads++;
                    hostBuffer.imageValid[i] = false;
                }
            } catch (const std::exception &e) {
                std::cerr << "Exception during FreeImage handling for " << filename << ": " << e.what() << std::endl;
                failedImageLoads++;
                hostBuffer.imageValid[i] = false;
                if (tempBmp) FreeImage_Unload(tempBmp);
                if (tempBmp32) FreeImage_Unload(tempBmp32);
                hostBuffer.imageBitmaps[i].reset();
            }
        }
        result.imagesSubmittedForProcessing += currentBatchActualSize;

        if (validImagesInBatchHost == 0) {
            std::cout << "  Skipping GPU processing for Batch " << batchIndex + 1 << " (no valid images loaded)." <<
                    std::endl;
            continue;
        }

        size_t validImagesSentToGpu = 0;
        int currentGpuIndex = 0;
        CUDA_CHECK(cudaMemsetAsync(d_hashResultBatch, 0, hashBatchTotalSize, 0));

        // Process valid images
        for (size_t i = 0; i < currentBatchActualSize; ++i) {
            if (!hostBuffer.imageValid[i]) continue;

            const auto &meta = hostBuffer.imageMetadata[i];
            const auto &bitmap = hostBuffer.imageBitmaps[i];
            int targetGpuSlot = currentGpuIndex;

            // Clean up previous texture resources for this slot
            if (d_textureObjects[targetGpuSlot] != 0) {
                cudaDestroyTextureObject(d_textureObjects[targetGpuSlot]);
                d_textureObjects[targetGpuSlot] = 0;
            }
            if (d_inputArrays[targetGpuSlot] != nullptr) {
                cudaFreeArray(d_inputArrays[targetGpuSlot]);
                d_inputArrays[targetGpuSlot] = nullptr;
            }

            // Allocate CUDA array for this image
            status = cudaMallocArray(&d_inputArrays[targetGpuSlot], &channelDesc, meta.width, meta.height);
            if (status != cudaSuccess) {
                std::cerr << "ERROR: Failed to allocate CUDA array for image " << hostBuffer.imagePaths[i]
                        << " (size: " << meta.width << "x" << meta.height << "): " << cudaGetErrorString(status) <<
                        std::endl;
                hostBuffer.imageValid[i] = false;
                failedKernelExecutions++;
                continue;
            }

            // Copy image data to CUDA array
            status = cudaMemcpy2DToArray(d_inputArrays[targetGpuSlot], 0, 0,
                                         FreeImage_GetBits(bitmap.get()), meta.pitch,
                                         meta.width * sizeof(uchar4), meta.height,
                                         cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                std::cerr << "ERROR: Failed to copy image data to CUDA array for " << hostBuffer.imagePaths[i]
                        << ": " << cudaGetErrorString(status) << std::endl;
                hostBuffer.imageValid[i] = false;
                failedKernelExecutions++;
                continue;
            }

            // Create texture object
            d_textureObjects[targetGpuSlot] =
                    createTextureObject(d_inputArrays[targetGpuSlot], meta.width, meta.height);
            if (d_textureObjects[targetGpuSlot] == 0) {
                std::cerr << "ERROR: Failed to create texture object for image " << hostBuffer.imagePaths[i] <<
                        std::endl;
                hostBuffer.imageValid[i] = false;
                failedKernelExecutions++;
                continue;
            }

            int gpuBatchOffset = currentGpuIndex * RESIZE_DIM * RESIZE_DIM;

            // Launch preprocessing kernel
            dim3 blockDimPreprocess(16, 16);
            dim3 gridDimPreprocess((RESIZE_DIM + blockDimPreprocess.x - 1) / blockDimPreprocess.x,
                                   (RESIZE_DIM + blockDimPreprocess.y - 1) / blockDimPreprocess.y);

            preprocessImageKernel_Texture<<<gridDimPreprocess, blockDimPreprocess, 0, 0>>>(
                d_textureObjects[targetGpuSlot], d_resizedGrayscaleBatch,
                meta.width, meta.height, gpuBatchOffset);

            dim3 blockDimDct(32, 1);
            dim3 gridDimDctRows((RESIZE_DIM + blockDimDct.x - 1) / blockDimDct.x, RESIZE_DIM);
            dct1d_kernel<<<gridDimDctRows, blockDimDct, 0, 0>>>(d_resizedGrayscaleBatch, d_dctIntermediateBatch,
                                                                RESIZE_DIM, RESIZE_DIM, true, gpuBatchOffset);

            dim3 gridDimDctCols((RESIZE_DIM + blockDimDct.x - 1) / blockDimDct.x, RESIZE_DIM);
            dct1d_kernel<<<gridDimDctCols, blockDimDct, 0, 0>>>(d_dctIntermediateBatch, d_dctResultBatch, RESIZE_DIM,
                                                                RESIZE_DIM, false, gpuBatchOffset);

            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr, "CUDA Kernel Error after launching kernels for image %s (index %zu): %s\n",
                        hostBuffer.imagePaths[i].c_str(), i, cudaGetErrorString(status));
                hostBuffer.imageValid[i] = false;
                failedKernelExecutions++;
                continue;
            }
            validImagesSentToGpu++;
            currentGpuIndex++;
        }

        // Hash computation kernel
        if (validImagesSentToGpu > 0) {
            dim3 blockDimHash(DCT_REDUCE_DIM, DCT_REDUCE_DIM);
            dim3 gridDimHash(validImagesSentToGpu, 1, 1);
            computeHashBitsKernel<<<gridDimHash, blockDimHash, sharedMemSize, 0>>>(
                d_dctResultBatch, d_hashResultBatch, validImagesSentToGpu);

            status = cudaGetLastError();
            if (status != cudaSuccess) {
                fprintf(stderr, "CUDA Kernel Error (computeHashBitsKernel Launch Check) for batch %zu: %s\n",
                        batchIndex, cudaGetErrorString(status));
                failedKernelExecutions += validImagesSentToGpu;
                validImagesSentToGpu = 0;
            } else {
                status = cudaMemcpyAsync(h_pinnedHashOutput, d_hashResultBatch,
                                         validImagesSentToGpu * hashBatchItemSize, cudaMemcpyDeviceToHost, 0);
                if (status != cudaSuccess) {
                    std::cerr << "ERROR: Failed D2H cudaMemcpyAsync for batch " << batchIndex << ": " <<
                            cudaGetErrorString(status) << std::endl;
                    failedKernelExecutions += validImagesSentToGpu;
                    validImagesSentToGpu = 0;
                }
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        // Collect results
        if (validImagesSentToGpu > 0) {
            size_t validHashIndex = 0;
            for (size_t i = 0; i < currentBatchActualSize; ++i) {
                if (hostBuffer.imageValid[i]) {
                    if (validHashIndex < validImagesSentToGpu) {
                        uint64_t currentHash = h_pinnedHashOutput[validHashIndex];
                        result.hashes.emplace_back(hostBuffer.imagePaths[i].string(), currentHash);
                        validHashIndex++;
                    } else {
                        std::cerr << "Warning: Mismatch retrieving hash for supposedly valid file (batch " << batchIndex
                                << "): " << hostBuffer.imagePaths[i] << ". Expected result but none available." <<
                                std::endl;
                        failedKernelExecutions++;
                    }
                }
            }
            if (validHashIndex != validImagesSentToGpu) {
                std::cerr << "Warning: Processed " << validHashIndex << " hashes, but expected " << validImagesSentToGpu
                        << " based on successful GPU submissions for batch " << batchIndex << std::endl;
            }
        }

        // Clean up texture resources for this batch
        for (size_t i = 0; i < MAX_BATCH_INSTANCES; ++i) {
            if (d_textureObjects[i] != 0) {
                cudaDestroyTextureObject(d_textureObjects[i]);
                d_textureObjects[i] = 0;
            }
            if (d_inputArrays[i] != nullptr) {
                cudaFreeArray(d_inputArrays[i]);
                d_inputArrays[i] = nullptr;
            }
        }
    } // End batch processing loop

    result.processingDuration = std::chrono::high_resolution_clock::now() - processingStartTime;
    std::cout << "Batch processing finished." << std::endl;

    std::sort(result.hashes.begin(), result.hashes.end());

    std::cout << "\n--- pHash Results (" << result.hashes.size() << " images) ---" << std::endl;
    for (const auto &res: result.hashes) {
        std::cout << res.first << ": 0x"
                << std::hex << std::setw(16) << std::setfill('0') << res.second << std::dec
                << std::endl;
    }

    std::cout << "\n--- Processing Performance ---" << std::endl;
    std::cout << "GPU Processing time (approx): " << result.processingDuration.count() << " seconds" << std::endl;
    if (!result.hashes.empty() && result.processingDuration.count() > 0.00001) {
        std::cout << "Throughput (successful):   " << (double) result.hashes.size() / result.processingDuration.count()
                << " images/sec" << std::endl;
    }
    if (result.imagesSubmittedForProcessing > 0 && result.processingDuration.count() > 0.00001) {
        std::cout << "Throughput (attempted):    " << (double) result.imagesSubmittedForProcessing / result.
                processingDuration.count()
                << " images/sec" << std::endl;
    }

compute_hashes_cleanup:
    std::cout << "Cleaning up computeHashes resources..." << std::endl;
    if (d_resizedGrayscaleBatch) cudaFree(d_resizedGrayscaleBatch);
    if (d_dctIntermediateBatch) cudaFree(d_dctIntermediateBatch);
    if (d_dctResultBatch) cudaFree(d_dctResultBatch);
    if (d_hashResultBatch) cudaFree(d_hashResultBatch);

    // Clean up texture
    for (size_t i = 0; i < MAX_BATCH_INSTANCES; ++i) {
        if (d_textureObjects[i] != 0) {
            cudaDestroyTextureObject(d_textureObjects[i]);
        }
        if (d_inputArrays[i] != nullptr) {
            cudaFreeArray(d_inputArrays[i]);
        }
    }

    if (h_pinnedHashOutput) cudaFreeHost(h_pinnedHashOutput);
    hostBuffer.clear();

    return result;
}
