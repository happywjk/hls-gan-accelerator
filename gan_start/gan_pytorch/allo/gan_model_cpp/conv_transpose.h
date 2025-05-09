#ifndef CONV_TRANSPOSE_H
#define CONV_TRANSPOSE_H

#include "gan_block_base.h"

// Declare the shared buffers
extern float shared_input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
extern float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
extern float shared_bias[MAX_CHANNELS];
extern float shared_conv_output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE];

// Process a single tile through convolution - now takes channel dimensions as parameters
void kernel_convolution_layer_tile(
    float input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    float bias[MAX_CHANNELS],
    float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
    int in_channels,
    int out_channels
);

// Load weights from global memory
void load_weights_tile(
    float *weight_data,
    float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    int in_channels,
    int out_channels
);

// Load bias from global memory
void load_bias_tile(
    float *bias_data,
    float bias[MAX_CHANNELS],
    int out_channels
);

#endif // CONV_TRANSPOSE_H