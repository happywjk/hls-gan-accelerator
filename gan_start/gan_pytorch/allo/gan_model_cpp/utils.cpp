#include "utils.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

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
) {
  const int input_channel_size = input_height * input_width;
  const int input_batch_size = in_channels * input_channel_size;

  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < in_channels; channel++) {
      for (int row = 0; row < KERNEL_SIZE; row++) {
        for (int col = 0; col < KERNEL_SIZE; col++) {
          // #pragma HLS INLINE OFF
          // #pragma HLS pipeline II=1 rewind
          
          // Calculate actual input position with padding adjustment
          int in_row_no_stride = row_start + row - KERNEL_SIZE + 1 + padding;
          int in_col_no_stride = col_start + col - KERNEL_SIZE + 1 + padding;
          int in_row = (in_row_no_stride) / stride;
          int in_col = (in_col_no_stride) / stride;
          
          bool is_orig_input_pos = (in_row_no_stride % stride == 0) && (in_col_no_stride % stride == 0);
          // Check if within input bounds
          if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width && is_orig_input_pos) {
            // Calculate index in the flattened input array
            int index = (batch * input_batch_size) + 
                        (channel * input_channel_size) + 
                        (in_row * input_width) + in_col;
            input_tile[batch][channel][row][col] = input_data[index];
          } else {
            // Zero padding for out-of-bounds
            input_tile[batch][channel][row][col] = 0.0f;
          }
        }
      }
    }
  }
}

// Store a single output tile to global memory
void store_output_tile(
  float *output_data,
  float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  int row,
  int col,
  int batch_size,
  int out_channels,
  int output_height,
  int output_width,
  int tile_height,
  int tile_width
) {
  // #pragma HLS INLINE OFF
  // #pragma HLS FUNCTION_INSTANTIATE variable=reuse_load_output
  const int output_channel_size = output_height * output_width;
  const int output_batch_size = out_channels * output_channel_size;

  // Check if the output tile position is within bounds
  if (row < output_height && col < output_width) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int channel = 0; channel < out_channels; channel++) {
        #pragma HLS pipeline II=1 rewind
        
        for (int tile_row = 0; tile_row < tile_height; tile_row++) {
          for (int tile_col = 0; tile_col < tile_width; tile_col++) {
            // Calculate position in output
            int out_row = row + tile_row;
            int out_col = col + tile_col;
            
            // Make sure we don't write outside the bounds
            if (out_row < output_height && out_col < output_width) {
              // Calculate index in the flattened output array
              int index = (batch * output_batch_size) + 
                          (channel * output_channel_size) + 
                          (out_row * output_width) + out_col;
              
              output_data[index] = output_tile[batch][channel][tile_row][tile_col];
            }
          }
        }
      }
    }
  }
}