#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <atomic>

#include <cuda_runtime.h>
#include <FreeImage.h>
#include <map>
#include <set>
#include <functional>

#include "phash/phash.cuh"
#include "constants_host_decl.h"
#include "lsh/lsh.cuh"


// FreeImage Initialization/Deinitialization
void FreeImageErrorHandler(FREE_IMAGE_FORMAT fif, const char *message) {
    std::cerr << "\n*** FreeImage Error ***\n";
    if (fif != FIF_UNKNOWN) {
        std::cerr << FreeImage_GetFormatFromFIF(fif) << " Format\n";
    }
    std::cerr << message << "\n***\n" << std::endl;
}

void initializeFreeImage() {
    FreeImage_Initialise();
    FreeImage_SetOutputMessage(FreeImageErrorHandler);
    std::cout << "FreeImage " << FreeImage_GetVersion() << " initialized." << std::endl;
    std::cout << FreeImage_GetCopyrightMessage() << std::endl;
}

void deinitializeFreeImage() {
    FreeImage_DeInitialise();
    std::cout << "FreeImage deinitialized." << std::endl;
}


bool initializeCUDA() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "CUDA Error: No CUDA-capable devices found or error querying devices: " << cudaGetErrorString(err)
                << std::endl;
        return false;
    }
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to set device 0: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, device);
    std::cout << "CUDA Initialized on device: " << deviceProp.name << std::endl;
    return true;
}


std::vector<std::filesystem::path> discoverImages(const std::filesystem::path &inputDir) {
    std::vector<std::filesystem::path> imagePaths;
    std::cout << "Scanning for images in: " << inputDir << std::endl;
    if (!std::filesystem::exists(inputDir) || !std::filesystem::is_directory(inputDir)) {
        std::cerr << "Error: Input directory does not exist or is not a directory: " << inputDir << std::endl;
        return imagePaths;
    }

    try {
        for (const auto &entry: std::filesystem::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                const std::filesystem::path &filePath = entry.path();
                std::string filePathStr = filePath.string(); // Need C-string for FreeImage
                const char *filePathCStr = filePathStr.c_str();

                FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filePathCStr, 0);
                if (format == FIF_UNKNOWN) {
                    format = FreeImage_GetFIFFromFilename(filePathCStr);
                }

                if (format != FIF_UNKNOWN && FreeImage_FIFSupportsReading(format)) {
                    imagePaths.push_back(filePath);
                }
            }
        }
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Filesystem error while scanning directory: " << e.what() << std::endl;
        imagePaths.clear();
    }

    std::cout << "Found " << imagePaths.size() << " potential image files." << std::endl;
    return imagePaths;
}


int main(int argc, char *argv[]) {
    auto totalStartTime = std::chrono::high_resolution_clock::now();
    int exitCode = 0;

    std::filesystem::path inputDirectory;
    std::vector<std::filesystem::path> allImagePaths;
    size_t totalImages = 0;

    std::vector<std::pair<std::string, uint64_t> > finalResults;
    std::atomic<size_t> failedImageLoads = 0;
    std::atomic<size_t> failedKernelExecutions = 0;
    size_t totalProcessedSuccessfully = 0;
    size_t imagesSubmittedForProcessing = 0;
    std::chrono::duration<double> processingDuration{};
    std::chrono::duration<double> totalDuration{};
    size_t unaccountedErrors = 0;

    std::chrono::system_clock::time_point totalEndTime;
    size_t accountedFor;
    size_t effectiveTotalImages;

    std::chrono::system_clock::time_point lshGroupingStartTime;

    // Initialization
    if (!initializeCUDA()) {
        exitCode = 1;
        goto cleanup;
    }
    initializeFreeImage();

    if (argc > 1) {
        inputDirectory = argv[1];
    } else {
        inputDirectory = DEFAULT_INPUT_PATH_CONST;
        std::cout << "No input directory provided, using default: " << inputDirectory << std::endl;
    }

    allImagePaths = discoverImages(inputDirectory);
    totalImages = allImagePaths.size();
    if (allImagePaths.empty() && argc <= 1) {
        std::cout << "No images in default path, using internal test data for LSH demo." << std::endl;
    }


    if (!allImagePaths.empty() || (allImagePaths.empty() && argc <= 1)) {
        ComputeHashesResult computationResult = computeHashes(allImagePaths, failedImageLoads, failedKernelExecutions);
        finalResults = std::move(computationResult.hashes);
        processingDuration = computationResult.processingDuration;
        imagesSubmittedForProcessing = computationResult.imagesSubmittedForProcessing;

        int target_number_of_new_duplicates = 40000; // how many to add (COMMENT OUT printing of the groups if used)

        // Start: Artificially inflate finalResults for testing LSH
        // if (!finalResults.empty() && target_number_of_new_duplicates > 0) {
        //     std::vector<std::pair<std::string, uint64_t> > newly_created_duplicates;
        //     std::string suffix_base = "_duplicate";
        //     int duplicate_filename_counter = 0;
        //
        //     const std::vector<std::pair<std::string, uint64_t> > &original_entries = finalResults;
        //     size_t num_original_entries = original_entries.size();
        //
        //     for (int i = 0; i < target_number_of_new_duplicates; ++i) {
        //         const auto &entry_to_duplicate = original_entries[i % num_original_entries];
        //
        //         std::filesystem::path originalPath(entry_to_duplicate.first);
        //         std::string newFilename = originalPath.stem().string() +
        //                                   suffix_base +
        //                                   std::to_string(duplicate_filename_counter++) +
        //                                   originalPath.extension().string();
        //         std::filesystem::path newPath = originalPath.parent_path() / newFilename;
        //         newly_created_duplicates.push_back({newPath.string(), entry_to_duplicate.second});
        //     }
        //
        //     if (!newly_created_duplicates.empty()) {
        //         finalResults.insert(finalResults.end(), newly_created_duplicates.begin(),
        //                             newly_created_duplicates.end());
        //         std::cout << "Artificially inflated finalResults by adding " << newly_created_duplicates.size()
        //                 << " new duplicated entries for LSH testing." << std::endl;
        //     }
        // } else if (target_number_of_new_duplicates > 0 && finalResults.empty()) {
        //     std::cout << "Cannot add duplicates: finalResults is empty." << std::endl;
        // }
        // End: Artificially inflate finalResults ---


        // Find Duplicates using LSH
        if (!finalResults.empty()) {
            std::cout << "\n--- Finding Duplicates using LSH ---" << std::endl;
            auto lshStartTime = std::chrono::high_resolution_clock::now();

            /* LSH Parameters */
            int numLshTables = 20; // L: Number of hash tables. Reduce for smaller memory usabe, but decreases recall.
            int lshKeySize = 8; // K: Bits per LSH key (e.g., 8 bits => 2^8=256 buckets per table max)
            int hammingDistThreshold = 2; // Max Hamming distance for duplicates (0 for exact)

            std::vector<LshDuplicateResult> duplicatePhashes = findDuplicatePhashes_LSH(
                finalResults,
                numLshTables,
                lshKeySize,
                hammingDistThreshold
            );

            auto lshEndTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> lshDuration = lshEndTime - lshStartTime;

            if (duplicatePhashes.empty()) {
                std::cout << "No duplicates found with Hamming distance <= " << hammingDistThreshold << "." <<
                        std::endl;
            } else {
                std::cout << "Found " << duplicatePhashes.size() << " duplicate/similar pairs." << std::endl;

                lshGroupingStartTime = std::chrono::high_resolution_clock::now();

                // Create a map from filename to index for efficient lookups
                std::map<std::string, int> fileToIndex;
                std::vector<std::string> indexToFile;
                int fileIndex = 0;

                // Build file index mapping
                for (const auto &dup: duplicatePhashes) {
                    if (fileToIndex.find(dup.file1) == fileToIndex.end()) {
                        fileToIndex[dup.file1] = fileIndex++;
                        indexToFile.push_back(dup.file1);
                    }
                    if (fileToIndex.find(dup.file2) == fileToIndex.end()) {
                        fileToIndex[dup.file2] = fileIndex++;
                        indexToFile.push_back(dup.file2);
                    }
                }

                // Union-Find parent array
                std::vector<int> parent(fileIndex);
                std::iota(parent.begin(), parent.end(), 0);

                std::function<int(int)> findRoot = [&](int x) -> int {
                    if (parent[x] != x) {
                        parent[x] = findRoot(parent[x]);
                    }
                    return parent[x];
                };

                // Union operation
                auto unite = [&](int x, int y) {
                    int px = findRoot(x), py = findRoot(y);
                    if (px != py) {
                        parent[px] = py;
                    }
                };

                // Build groups by uniting similar files
                for (const auto &dup: duplicatePhashes) {
                    int idx1 = fileToIndex[dup.file1];
                    int idx2 = fileToIndex[dup.file2];
                    unite(idx1, idx2);
                }

                // Group files by their root parent
                std::map<int, std::vector<std::string> > groups;
                for (int i = 0; i < fileIndex; ++i) {
                    int root = findRoot(i);
                    groups[root].push_back(indexToFile[i]);
                }

                std::chrono::duration<double> groupingDuration =
                        std::chrono::high_resolution_clock::now() - lshGroupingStartTime;

                std::cout << "Grouping took " << groupingDuration.count() << " seconds." << std::endl;

                std::cout << "\n--- Duplicate/Similar Image Groups ---" << std::endl;
                int groupNum = 1;

                for (const auto &group: groups) {
                    if (group.second.size() > 1) {
                        std::cout << "Group " << groupNum << " (" << group.second.size() << " files):" << std::endl;

                        // Show all files in this group
                        for (const auto &file: group.second) {
                            uint64_t fileHash = 0;
                            for (const auto &result: finalResults) {
                                if (result.first == file) {
                                    fileHash = result.second;
                                    break;
                                }
                            }
                            std::cout << "  - " << file << " (hash: 0x" << std::hex << fileHash << std::dec << ")" <<
                                    std::endl;
                        }

                        // Show pairwise distances within this group
                        std::cout << "  Relationships:" << std::endl;
                        for (const auto &dup: duplicatePhashes) {
                            bool file1InGroup = std::find(group.second.begin(), group.second.end(), dup.file1) != group.
                                                second.end();
                            bool file2InGroup = std::find(group.second.begin(), group.second.end(), dup.file2) != group.
                                                second.end();

                            if (file1InGroup && file2InGroup) {
                                std::cout << "    " << dup.file1 << " ↔ " << dup.file2
                                        << " (distance: " << dup.distance << ")" << std::endl;
                            }
                        }

                        std::cout << std::endl;
                        groupNum++;
                    }
                }

                std::cout << "Total duplicate groups found: " << (groupNum - 1) << std::endl;
            }
            std::cout << "LSH processing time: " << lshDuration.count() << " seconds" << std::endl;
        }
    } else {
        std::cout << "Skipping hash computation as no images were found (and not using test data)." << std::endl;
        processingDuration = std::chrono::duration<double>(0);
    }

    // Summary
    totalEndTime = std::chrono::high_resolution_clock::now();
    totalDuration = totalEndTime - totalStartTime;
    totalProcessedSuccessfully = finalResults.size();

    accountedFor = failedImageLoads.load() + totalProcessedSuccessfully + failedKernelExecutions.load();
    unaccountedErrors = 0;
    effectiveTotalImages = (totalImages == 0 && !finalResults.empty()) ? finalResults.size() : totalImages;


    if (effectiveTotalImages > accountedFor) {
        unaccountedErrors = effectiveTotalImages - accountedFor;
    } else if (accountedFor > effectiveTotalImages && effectiveTotalImages > 0) {
        std::cerr << "Warning: Accounted for more images (" << accountedFor
                << ") than initially found/processed (" << effectiveTotalImages << "). Error in counting logic." <<
                std::endl;
    }

    std::cout << "\n--- Overall Summary ---" << std::endl;
    std::cout << "Total images considered:   " << effectiveTotalImages << std::endl;
    std::cout << "Images processed for hash: " << totalProcessedSuccessfully << std::endl;
    std::cout << "Image load/convert errors: " << failedImageLoads.load() << std::endl;
    std::cout << "GPU processing errors (pHash): " << failedKernelExecutions.load() << std::endl;
    std::cout << "Other/Unaccounted errors:  " << unaccountedErrors << std::endl;
    std::cout << "Total execution time:      " << totalDuration.count() << " seconds" << std::endl;

cleanup:
    std::cout << "Cleaning up main resources..." << std::endl;
    deinitializeFreeImage();
    if (exitCode == 0) {
        cudaError_t resetStatus = cudaDeviceReset();
        if (resetStatus != cudaSuccess) {
            std::cerr << "Warning: cudaDeviceReset failed with error: " << cudaGetErrorString(resetStatus) << std::endl;
            if (exitCode == 0) exitCode = 2;
        }
    }

    std::cout << "Cleanup complete. Exiting with code: " << exitCode << std::endl;
    return exitCode;
}
