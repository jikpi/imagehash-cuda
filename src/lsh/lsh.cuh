#ifndef LSH_CUH
#define LSH_CUH

#include <vector>
#include <string>
#include <cstdint>
#include <utility>

// Struct to hold LSH duplicate results
struct LshDuplicateResult {
    std::string file1;
    uint64_t hash1;
    std::string file2;
    uint64_t hash2;
    int distance;
};

// Find duplicate or near-duplicate phashes using Locality Sensitive Hashing.
std::vector<LshDuplicateResult> findDuplicatePhashes_LSH(
    const std::vector<std::pair<std::string, uint64_t>>& allPhashesInput,
    int numTables,
    int keySize,
    int hammingThreshold
);

// CUDA Kernels

// Kernel to generate LSH keys for each pHash for multiple tables.
__global__ void generateLshKeysKernel(
    const uint64_t* d_phashes,          // Input: array of phashes on device
    unsigned int* d_lshKeys,          // Output: flat array of LSH keys [phashIdx * numTables + tableIdx]
    const unsigned int* d_bitSelectionIndices, // Input: indices of bits to select for keys [tableIdx * keySize + k_idx]
    int numPhashes,
    int numTables,
    int keySize
);

// Kernel to compute Hamming distances for pairs of phashes.
__global__ void batchHammingDistanceKernel(
    const uint64_t* d_allPhashes,       // Input: array of all phashes on device
    const unsigned int* d_queryIndices,   // Input: indices into d_allPhashes for the first hash in a pair
    const unsigned int* d_candidateIndices, // Input: indices into d_allPhashes for the second hash in a pair
    int* d_distances,                   // Output: array of Hamming distances for each pair
    int numPairs                        // Number of pairs to compare
);


// Device function to calculate Hamming distance between two 64-bit integers.
__device__ int calculateHammingDistance(uint64_t h1, uint64_t h2);

#endif // LSH_CUH