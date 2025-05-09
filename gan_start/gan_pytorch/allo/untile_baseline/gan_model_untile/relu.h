#ifndef RELU_H
#define RELU_H

#include "gan_block_base.h"

// Process a single tile through ReLU
void kernel_relu_layer(
    float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
    float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
    int batch_size,
    int channels,
    int height,
    int width
  ) ;

#endif // RELU_H