#ifndef RELU_H
#define RELU_H

#include "gan_block_base.h"

// Process a single tile through ReLU
void kernel_relu_layer_tile(
    float input_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
    float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
    int batch_size,
    int channels,
    int tile_height,
    int tile_width
);

#endif // RELU_H