#ifndef CONV_TRANSPOSE_H
#define CONV_TRANSPOSE_H

#include "gan_block_base.h"



// Process a single tile through convolution - now takes channel dimensions as parameters
void kernel_convolution_transpose_layer(
    float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
    float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    float bias[MAX_CHANNELS],
    float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
    int in_channels,
    int out_channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int stride,
    int padding
  );

// Load weights from global memory
void load_weights(
    float *weight_data,
    float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    int in_channels,
    int out_channels
);

// Load bias from global memory
void load_bias(
    float *bias_data,
    float bias[MAX_CHANNELS],
    int out_channels
);

#endif // CONV_TRANSPOSE_H