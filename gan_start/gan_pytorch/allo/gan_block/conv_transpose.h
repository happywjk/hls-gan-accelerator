#ifndef CONV_TRANSPOSE_H
#define CONV_TRANSPOSE_H

#include "gan_block.h"

// Process a single tile through convolution
void kernel_convolution_layer_tile(
    float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    float bias[OUT_CHANNELS],
    float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1]
);

// Load weights from global memory
void load_weights_tile(
    float *weight_data,
    float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE]
);

// Load bias from global memory
void load_bias_tile(
    float *bias_data,
    float bias[OUT_CHANNELS]
);

#endif // CONV_TRANSPOSE_H