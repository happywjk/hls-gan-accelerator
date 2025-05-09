#ifndef UTILS_H
#define UTILS_H

#include "gan_block_base.h" 

void load_input_tile(
    float *input_data,
    float input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    int row_start,
    int col_start,
    int in_channels,
    int input_height,
    int input_width,
    int padding,
    int stride
  );

// Store an output tile to global memory with parameters for dimensions
void store_output_tile(
    float *output_data,
    float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
    int row,
    int col,
    int batch_size,
    int out_channels,
    int output_height,
    int output_width,
    int tile_height = 1,
    int tile_width = 1
);

#endif // UTILS_H