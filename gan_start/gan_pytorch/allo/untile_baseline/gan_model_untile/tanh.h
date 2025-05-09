#ifndef TANH_H
#define TANH_H

#include "gan_block_base.h"

// Apply tanh activation to a tile of data
void kernel_tanh_layer(
  float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int batch_size,
  int channels,
  int height,
  int width
);

#endif // TANH_H