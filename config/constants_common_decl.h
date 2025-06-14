#ifndef CONSTANTS_COMMON_DECL_H
#define CONSTANTS_COMMON_DECL_H

constexpr float PI = 3.1415926535f;

constexpr int RESIZE_DIM = 32; // Resize image to 32x32
constexpr int DCT_REDUCE_DIM = 8; // Reduce DCT to 8x8 low-frequency block
constexpr int HASH_BITS = DCT_REDUCE_DIM * DCT_REDUCE_DIM; // 64 bits

#endif // CONSTANTS_COMMON_DECL_H