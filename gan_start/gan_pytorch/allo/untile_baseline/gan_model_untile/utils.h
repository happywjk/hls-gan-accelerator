#ifndef UTILS_H
#define UTILS_H

#include "gan_block_base.h" 

void load_input(
  float *input_data,
  float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int in_channels,
  int input_height,
  int input_width,
  int output_height,
  int output_width,
  int output_padding,
  int padding,
  int stride
) ;

// Store an output tile to global memory with parameters for dimensions
void store_output(
  float *output_data,
  float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int batch_size,
  int out_channels,
  int output_height,
  int output_width
);

#endif // UTILS_H