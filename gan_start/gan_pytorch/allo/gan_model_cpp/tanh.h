#ifndef TANH_H
#define TANH_H

#include "gan_block_base.h"

// Apply tanh activation to a tile of data
void kernel_tanh_layer_tile(
  float input_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  int batch_size,
  int channels,
  int height,
  int width
);

#endif // TANH_H