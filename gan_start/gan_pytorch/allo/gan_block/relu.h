#ifndef RELU_H
#define RELU_H

#include "gan_block.h"

// Apply ReLU activation to a tile
void kernel_relu_layer_tile(
    float input_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
    float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1]
);

#endif // RELU_H