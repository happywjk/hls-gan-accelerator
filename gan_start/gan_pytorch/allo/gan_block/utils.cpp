#include "utils.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Load a single input tile from global memory with proper padding
void load_input_tile(
  float *input_data,
  float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  int row_start,
  int col_start
) {
  const int input_channel_size = INPUT_HEIGHT * INPUT_WIDTH;
  const int input_batch_size = IN_CHANNELS * input_channel_size;

  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < IN_CHANNELS; channel++) {
      for (int row = 0; row < KERNEL_SIZE; row++) {
        for (int col = 0; col < KERNEL_SIZE; col++) {
          #pragma HLS pipeline II=1 rewind
          
          // Calculate actual input position with padding adjustment
          int in_row_no_stride = row_start + row - KERNEL_SIZE + 1 + PADDING;
          int in_col_no_stride = col_start + col - KERNEL_SIZE + 1 + PADDING;
          int in_row = (in_row_no_stride) / stride;
          int in_col = (in_col_no_stride) / stride;
          
          bool is_orig_input_pos = (in_row_no_stride % stride == 0) && (in_col_no_stride % stride == 0);
          // Check if within input bounds
          if (in_row >= 0 && in_row < INPUT_HEIGHT && in_col >= 0 && in_col < INPUT_WIDTH && is_orig_input_pos) {
            // Calculate index in the flattened input array
            int index = (batch * input_batch_size) + 
                        (channel * input_channel_size) + 
                        (in_row * INPUT_WIDTH) + in_col;
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
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
  int row,
  int col
) {
  const int output_channel_size = OUTPUT_HEIGHT * OUTPUT_WIDTH;
  const int output_batch_size = OUT_CHANNELS * output_channel_size;

  // Check if the output tile position is within bounds
  if (row < OUTPUT_HEIGHT && col < OUTPUT_WIDTH) {
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
      for (int channel = 0; channel < OUT_CHANNELS; channel++) {
        #pragma HLS pipeline II=1 rewind
        
        // Calculate index in the flattened output array
        int index = (batch * output_batch_size) + 
                    (channel * output_channel_size) + 
                    (row * OUTPUT_WIDTH) + col;
        output_data[index] = output_tile[batch][channel][0][0];
      }
    }
  }
}

