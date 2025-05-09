#ifndef UTILS_H
#define UTILS_H

#include "gan_block.h"

// Load an input tile from global memory
// void load_input_tile(
//     float *input_data,
//     float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
//     int row_start,
//     int col_start
// );

// Store an output tile to global memory
void store_output_tile(
    float *output_data,
    float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
    int row,
    int col
);

#endif // UTILS_H