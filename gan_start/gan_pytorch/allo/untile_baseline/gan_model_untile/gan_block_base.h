#ifndef GAN_BLOCK_BASE_H
#define GAN_BLOCK_BASE_H

// Common constants and types
const int BATCH_SIZE = 8;
const int KERNEL_SIZE = 4;
const int stride = 2;
const int UNROLL_FACTOR = 1;

// These are declared but not defined - will be overridden
const int MAX_CHANNELS = 256;
const int MAX_INPUT_WIDTH = 64;
const int MAX_INPUT_HEIGHT = 64;
const int MAX_OUTPUT_WIDTH = 132; //consider padding
const int MAX_OUTPUT_HEIGHT = 132;
#endif // GAN_BLOCK_BASE_H