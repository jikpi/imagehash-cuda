//
// Created by skl on 2025-06-13.
//

#ifndef PHASH_H
#define PHASH_H
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <atomic>
#include "../helpers/FreeImageBitmap.h"

// computeHashes return value
struct ComputeHashesResult {
    std::vector<std::pair<std::string, uint64_t> > hashes;
    std::chrono::duration<double> processingDuration{};
    size_t imagesSubmittedForProcessing{};
};

struct ImageMetadata {
    unsigned int width = 0;
    unsigned int height = 0;
    unsigned int pitch = 0;
    size_t rawSize = 0; // Size in bytes of the 32-bit BGRA data
};

// Single Host buffer for image data
struct HostBatchBuffer {
    std::vector<FreeImageBitmap> imageBitmaps;
    std::vector<std::filesystem::path> imagePaths;
    std::vector<ImageMetadata> imageMetadata;
    std::vector<bool> imageValid; // Track successful load/convert
    size_t batchSize = 0; // Store the actual size of the batch loaded here

    HostBatchBuffer();
    void clear();
};

// Function declaration
ComputeHashesResult computeHashes(
    const std::vector<std::filesystem::path> &allImagePaths,
    std::atomic<size_t> &failedImageLoads, // Passed by reference to update main's counter
    std::atomic<size_t> &failedKernelExecutions // Passed by reference
);

#endif //PHASH_H