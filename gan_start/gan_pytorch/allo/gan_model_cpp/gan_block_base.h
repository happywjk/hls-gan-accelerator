#ifndef GAN_BLOCK_BASE_H
#define GAN_BLOCK_BASE_H

// Common constants and types
const int BATCH_SIZE = 8;
const int KERNEL_SIZE = 4;
const int stride = 2;
const int UNROLL_FACTOR = 1;

// These are declared but not defined - will be overridden
const int MAX_CHANNELS = 256;
const int MAX_TILE_SIZE = 1;
#endif // GAN_BLOCK_BASE_H