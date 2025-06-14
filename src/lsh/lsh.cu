//
// Created by skl on 2025-06-13.
//

#include "lsh.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <cstdio>

#define CUDA_CHECK_LSH(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        } \
    } while(0)

// Device Code

__device__ int calculateHammingDistance(uint64_t h1, uint64_t h2) {
    uint64_t xor_result = h1 ^ h2;
    int distance = 0;
    while (xor_result > 0) {
        xor_result &= (xor_result - 1);
        distance++;
    }
    return distance;
}

__global__ void generateLshKeysKernel(
    const uint64_t *d_phashes,
    unsigned int *d_lshKeys,
    const unsigned int *d_bitSelectionIndices,
    int numPhashes,
    int numTables,
    int keySize
) {
    int phashIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (phashIdx >= numPhashes) {
        return;
    }

    uint64_t currentPhash = d_phashes[phashIdx];

    for (int tableIdx = 0; tableIdx < numTables; ++tableIdx) {
        unsigned int currentLshKey = 0;
        for (int k = 0; k < keySize; ++k) {
            // Get the bit position (0-63) for the k-th bit of the LSH key in table tableIdx
            unsigned int bitPos = d_bitSelectionIndices[tableIdx * keySize + k];
            // If the selected bit in the pHash is set, set the k-th bit in the LSH key
            if ((currentPhash >> bitPos) & 1ULL) {
                currentLshKey |= (1U << k);
            }
        }
        d_lshKeys[phashIdx * numTables + tableIdx] = currentLshKey;
    }
}

__global__ void batchHammingDistanceKernel(
    const uint64_t *d_allPhashes,
    const unsigned int *d_queryIndices,
    const unsigned int *d_candidateIndices,
    int *d_distances,
    int numPairs
) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) {
        return;
    }

    unsigned int queryIdx = d_queryIndices[pairIdx];
    unsigned int candidateIdx = d_candidateIndices[pairIdx];

    uint64_t hash1 = d_allPhashes[queryIdx];
    uint64_t hash2 = d_allPhashes[candidateIdx];

    d_distances[pairIdx] = calculateHammingDistance(hash1, hash2);
}

// Host

std::vector<LshDuplicateResult> findDuplicatePhashes_LSH(
    const std::vector<std::pair<std::string, uint64_t> > &allPhashesInput,
    int numTables,
    int keySize,
    int hammingThreshold
) {
    std::vector<LshDuplicateResult> duplicateResults;
    if (allPhashesInput.empty()) {
        return duplicateResults;
    }

    int numPhashes = static_cast<int>(allPhashesInput.size());

    // 1. Prepare pHash data (extract uint64_t for GPU)
    std::vector<uint64_t> h_phashes(numPhashes);
    for (int i = 0; i < numPhashes; ++i) {
        h_phashes[i] = allPhashesInput[i].second;
    }

    std::cout << "1 Prepared pHash data" << std::endl;

    // 2. Allocate GPU memory
    uint64_t *d_phashes = nullptr;
    unsigned int *d_lshKeys = nullptr; // LSH keys: numPhashes * numTables
    unsigned int *d_bitSelectionIndices = nullptr; // bit positions for LSH: numTables * keySize

    CUDA_CHECK_LSH(cudaMalloc(&d_phashes, numPhashes * sizeof(uint64_t)));
    CUDA_CHECK_LSH(cudaMalloc(&d_lshKeys, numPhashes * numTables * sizeof(unsigned int)));
    CUDA_CHECK_LSH(cudaMalloc(&d_bitSelectionIndices, numTables * keySize * sizeof(unsigned int)));

    std::cout << "2 Allocated GPU memory" << std::endl;

    // 3. Copy phashes to GPU
    CUDA_CHECK_LSH(cudaMemcpy(d_phashes, h_phashes.data(), numPhashes * sizeof(uint64_t), cudaMemcpyHostToDevice));

    std::cout << "3 Copied pHash data to GPU" << std::endl;

    // 4. Generate random bit selection indices on CPU for LSH keys
    // For each table, select 'keySize' unique bit positions (0-63)
    std::vector<unsigned int> h_bitSelectionIndices(numTables * keySize);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned int> dist(0, 63);

    for (int t = 0; t < numTables; ++t) {
        std::set<unsigned int> usedBitsForKey;
        for (int k = 0; k < keySize; ++k) {
            unsigned int bitPos;
            do {
                bitPos = dist(rng);
            } while (usedBitsForKey.count(bitPos));
            h_bitSelectionIndices[t * keySize + k] = bitPos;
            usedBitsForKey.insert(bitPos);
        }
    }
    CUDA_CHECK_LSH(
        cudaMemcpy(d_bitSelectionIndices, h_bitSelectionIndices.data(), numTables * keySize * sizeof(unsigned int),
            cudaMemcpyHostToDevice));

    std::cout << "4 Generated random bit selection indices for LSH keys" << std::endl;

    // 5. Launch LSH key generation kernel
    int threadsPerBlock = 256;
    int blocksPerGridLsh = (numPhashes + threadsPerBlock - 1) / threadsPerBlock;
    generateLshKeysKernel<<<blocksPerGridLsh, threadsPerBlock>>>(
        d_phashes, d_lshKeys, d_bitSelectionIndices, numPhashes, numTables, keySize
    );
    CUDA_CHECK_LSH(cudaGetLastError()); // Check for kernel launch errors
    CUDA_CHECK_LSH(cudaDeviceSynchronize()); // Wait for kernel to complete

    std::cout << "5 Generated LSH keys" << std::endl;

    // 6. Copy LSH keys back to CPU
    std::vector<unsigned int> h_lshKeys(numPhashes * numTables);
    CUDA_CHECK_LSH(
        cudaMemcpy(h_lshKeys.data(), d_lshKeys, numPhashes * numTables * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::cout << "6 Generated LSH keys" << std::endl;

    // 7. Build hash tables (buckets) on CPU
    // For each table, map LSH key to a vector of pHash indices that produced that key
    std::vector<std::map<unsigned int, std::vector<int> > > hashTables(numTables);
    for (int i = 0; i < numPhashes; ++i) {
        // For each pHash
        for (int t = 0; t < numTables; ++t) {
            // For each LSH table
            unsigned int key = h_lshKeys[i * numTables + t];
            hashTables[t][key].push_back(i); // Add pHash index 'i' to the bucket for 'key' in table 't'
        }
    }

    std::cout << "7 Built hash tables for LSH keys" << std::endl;

    // 8. Generate candidate pairs for Hamming distance calculation
    // A set stores unique pairs (idx1, idx2)
    std::set<std::pair<int, int> > candidatePairs;
    for (int t = 0; t < numTables; ++t) {
        // For each LSH table
        for (const auto &bucketEntry: hashTables[t]) {
            // For each bucket in the table
            const std::vector<int> &pHashIndicesInBucket = bucketEntry.second;
            if (pHashIndicesInBucket.size() > 1) {
                // If bucket has more than one pHash, they are candidates
                for (size_t i = 0; i < pHashIndicesInBucket.size(); ++i) {
                    for (size_t j = i + 1; j < pHashIndicesInBucket.size(); ++j) {
                        int idx1 = pHashIndicesInBucket[i];
                        int idx2 = pHashIndicesInBucket[j];
                        candidatePairs.insert(std::minmax(idx1, idx2)); // Insert sorted pair
                    }
                }
            }
        }
    }

    std::cout << "8 Generated candidate pairs for Hamming distance calculation" << std::endl;

    if (candidatePairs.empty()) {
        std::cout << "LSH: No candidate pairs found after bucketing." << std::endl;
        CUDA_CHECK_LSH(cudaFree(d_phashes));
        CUDA_CHECK_LSH(cudaFree(d_lshKeys));
        CUDA_CHECK_LSH(cudaFree(d_bitSelectionIndices));
        return duplicateResults;
    }

    std::cout << "8 Found " << candidatePairs.size() << " candidate pairs for Hamming distance calculation" <<
            std::endl;

    // 9. Prepare data for batch Hamming distance kernel
    int numCandidatePairs = static_cast<int>(candidatePairs.size());
    std::vector<unsigned int> h_queryIndices(numCandidatePairs);
    std::vector<unsigned int> h_candidateIndices(numCandidatePairs);
    int pairIdx = 0;
    for (const auto &pair: candidatePairs) {
        h_queryIndices[pairIdx] = pair.first;
        h_candidateIndices[pairIdx] = pair.second;
        pairIdx++;
    }

    unsigned int *d_queryIndices_hd = nullptr;
    unsigned int *d_candidateIndices_hd = nullptr;
    int *d_distances = nullptr;

    CUDA_CHECK_LSH(cudaMalloc(&d_queryIndices_hd, numCandidatePairs * sizeof(unsigned int)));
    CUDA_CHECK_LSH(cudaMalloc(&d_candidateIndices_hd, numCandidatePairs * sizeof(unsigned int)));
    CUDA_CHECK_LSH(cudaMalloc(&d_distances, numCandidatePairs * sizeof(int)));

    CUDA_CHECK_LSH(
        cudaMemcpy(d_queryIndices_hd, h_queryIndices.data(), numCandidatePairs * sizeof(unsigned int),
            cudaMemcpyHostToDevice));
    CUDA_CHECK_LSH(
        cudaMemcpy(d_candidateIndices_hd, h_candidateIndices.data(), numCandidatePairs * sizeof(unsigned int),
            cudaMemcpyHostToDevice));

    std::cout << "9 Prepared data for batch Hamming distance kernel" << std::endl;

    // 10. Launch batch Hamming distance kernel
    int blocksPerGridHd = (numCandidatePairs + threadsPerBlock - 1) / threadsPerBlock;
    batchHammingDistanceKernel<<<blocksPerGridHd, threadsPerBlock>>>(
        d_phashes, // original d_phashes array
        d_queryIndices_hd,
        d_candidateIndices_hd,
        d_distances,
        numCandidatePairs
    );
    CUDA_CHECK_LSH(cudaGetLastError());
    CUDA_CHECK_LSH(cudaDeviceSynchronize());

    std::cout << "10 Launched batch Hamming distance kernel" << std::endl;

    // 11. Copy distances back to CPU
    std::vector<int> h_distances(numCandidatePairs);
    CUDA_CHECK_LSH(cudaMemcpy(h_distances.data(), d_distances, numCandidatePairs * sizeof(int), cudaMemcpyDeviceToHost))
    ;

    std::cout << "11 Copied Hamming distances back to CPU" << std::endl;

    // 12. Filter results based on Hamming threshold and populate LshDuplicateResult
    pairIdx = 0;
    for (const auto &pair: candidatePairs) {
        // Iterate in the same order as h_distances was populated
        if (h_distances[pairIdx] <= hammingThreshold) {
            LshDuplicateResult res;
            res.file1 = allPhashesInput[pair.first].first; // original index
            res.hash1 = allPhashesInput[pair.first].second;
            res.file2 = allPhashesInput[pair.second].first; // original index
            res.hash2 = allPhashesInput[pair.second].second;
            res.distance = h_distances[pairIdx];
            duplicateResults.push_back(res);
        }
        pairIdx++;
    }

    std::cout << "12 Filtered results based on Hamming threshold" << std::endl;

    // 13. Cleanup GPU memory
    CUDA_CHECK_LSH(cudaFree(d_phashes));
    CUDA_CHECK_LSH(cudaFree(d_lshKeys));
    CUDA_CHECK_LSH(cudaFree(d_bitSelectionIndices));
    CUDA_CHECK_LSH(cudaFree(d_queryIndices_hd));
    CUDA_CHECK_LSH(cudaFree(d_candidateIndices_hd));
    CUDA_CHECK_LSH(cudaFree(d_distances));

    return duplicateResults;
}
