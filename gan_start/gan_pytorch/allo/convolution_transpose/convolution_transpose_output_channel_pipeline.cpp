#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Global parameters
const int BATCH_SIZE = 8;
const int IN_CHANNELS = 32;
const int OUT_CHANNELS = 16;
const int KERNEL_SIZE = 4;
const int INPUT_HEIGHT = 32;
const int INPUT_WIDTH = 32;
const int PADDING = 1;
const int stride = 2;
const int OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) * stride + KERNEL_SIZE - 2*PADDING; 
const int OUTPUT_WIDTH = (INPUT_WIDTH - 1) * stride + KERNEL_SIZE - 2*PADDING;
const int UNROLL_FACTOR = 1;

extern "C" {
// Process a single tile of the input
void kernel_convolution_layer_tile(
  float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float bias[OUT_CHANNELS],
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1]
) {
  // This kernel processes a tile and produces a 1x1 output tile
  #pragma HLS inline off
  #pragma HLS array_partition variable=weights complete dim=3  // kernel_row dimension
  #pragma HLS array_partition variable=weights complete dim=4  // kernel_col dimension
  #pragma HLS array_partition variable=input_tile complete dim=3  // kernel_row dimension
  #pragma HLS array_partition variable=input_tile complete dim=4 

  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++) {
      #pragma HLS pipeline II=1
      float sum = bias[out_channel];
      for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++) {
        for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
          #pragma HLS unroll
          for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
            #pragma HLS unroll

            sum += weights[in_channel][out_channel][kernel_row][kernel_col] * 
                   input_tile[batch][in_channel][kernel_row][kernel_col];
          }
        }
      }
      
      output_tile[batch][out_channel][0][0] = sum;
    }
  }
}
// Load a single input tile from global memory with proper padding
void load_input_tile(
  float *input_data,
  float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  int row_start,
  int col_start
) {
  #pragma HLS inline off 
  const int input_channel_size = INPUT_HEIGHT * INPUT_WIDTH;
  const int input_batch_size = IN_CHANNELS * input_channel_size;

  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < IN_CHANNELS; channel++) {
      for (int row = 0; row < KERNEL_SIZE; row++) {
        for (int col = 0; col < KERNEL_SIZE; col++) {
          #pragma HLS pipeline II=1 rewind
          
          // Calculate actual input position with padding adjustment
          int in_row_no_stride = row_start + row - KERNEL_SIZE +1 + PADDING;
          int in_col_no_stride = col_start + col - KERNEL_SIZE +1 + PADDING;
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

// Load weights for processing
void load_weights_tile(
  float *weight_data,
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE]
) {
  const int weight_channel_size = KERNEL_SIZE * KERNEL_SIZE;
  const int weight_out_channel_size = OUT_CHANNELS * weight_channel_size;

  for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++) {
    for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++) {
      for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
        for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
          #pragma HLS pipeline II=1 rewind
          int index = (in_channel * weight_out_channel_size) + 
                      (out_channel * weight_channel_size) + 
                      (kernel_row * KERNEL_SIZE) + kernel_col;
          int flipped_row = KERNEL_SIZE - 1 - kernel_row;
          int flipped_col = KERNEL_SIZE - 1 - kernel_col;
          weights[in_channel][out_channel][flipped_row][flipped_col] = weight_data[index];
        }
      }
    }
  }
}

// Load bias data
void load_bias_tile(
  float *bias_data,
  float bias[OUT_CHANNELS]
) {
  for (int i = 0; i < OUT_CHANNELS; i++) {
    #pragma HLS pipeline II=1 rewind
    bias[i] = bias_data[i];
  }
}

// Store a single output tile to global memory
void store_output_tile(
  float *output_data,
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
  int row,
  int col
) {
  #pragma HLS inline off
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

void top(
  float *input_data,
  float *weight_data,
  float *bias_data,
  float *output_data
) {
  #pragma HLS interface m_axi port=input_data offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=weight_data offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=bias_data offset=slave bundle=gmem2
  #pragma HLS interface m_axi port=output_data offset=slave bundle=gmem3
  
  // Allocate on-chip buffers (BRAM)
  float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float bias[OUT_CHANNELS];
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1];
  
  
  // Load weights and biases once (assuming they fit in BRAM)
  load_weights_tile(weight_data, weights);
  load_bias_tile(bias_data, bias);
  
  // Process each output position
  for (int out_row = 0; out_row < OUTPUT_HEIGHT; out_row++) { 
    for (int out_col = 0; out_col < OUTPUT_WIDTH; out_col++) {
      // Calculate corresponding input position
      
      // 1. Load the input tile for this position
      load_input_tile(input_data, input_tile, out_row, out_col);
      
      // 2. Process the tile - launch kernel
      kernel_convolution_layer_tile(input_tile, weights, bias, output_tile);
      
      // 3. Store the output tile
      store_output_tile(output_data, output_tile, out_row, out_col);
    }
  }
}

}