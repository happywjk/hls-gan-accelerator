#include "conv_transpose.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Reference to the shared buffers (defined in gan_model.cpp)
extern float shared_input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
extern float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
extern float shared_bias[MAX_CHANNELS];
extern float shared_conv_output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE];

// Process a single tile of the input - with variable channel dimensions
void kernel_convolution_layer_tile(
  float input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float bias[MAX_CHANNELS],
  float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  int in_channels,
  int out_channels
) {
  // This kernel processes a tile and produces a tile output
  // #pragma HLS INLINE OFF
  // #pragma HLS FUNCTION_INSTANTIATE variable=reuse_conv_instance
  
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int out_channel = 0; out_channel < out_channels; out_channel++) {
      #pragma HLS pipeline II=1
      float sum = bias[out_channel];
      
      for (int in_channel = 0; in_channel < in_channels; in_channel++) {
        #pragma HLS unroll
        for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
          #pragma HLS unroll 
          for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
            #pragma HLS unroll 
            sum += weights[in_channel][out_channel][kernel_row][kernel_col] * 
                   input_tile[batch][in_channel][kernel_row][kernel_col];
          }
        }
      }
      
      // Assuming we're generating 1x1 outputs for each tile, adjust as needed
      output_tile[batch][out_channel][0][0] = sum;
    }
  }
}

// Load weights for processing - with variable channel dimensions
void load_weights_tile(
  float *weight_data,
  float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  int in_channels,
  int out_channels
) {
  // #pragma HLS INLINE OFF
  // #pragma HLS FUNCTION_INSTANTIATE variable=reuse_load_weight
  const int weight_channel_size = KERNEL_SIZE * KERNEL_SIZE;
  const int weight_out_channel_size = out_channels * weight_channel_size;

  for (int in_channel = 0; in_channel < in_channels; in_channel++) {
    for (int out_channel = 0; out_channel < out_channels; out_channel++) {
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

// Load bias data - with variable channel dimensions
void load_bias_tile(
  float *bias_data,
  float bias[MAX_CHANNELS],
  int out_channels
) {
  // #pragma HLS INLINE OFF
  // #pragma HLS FUNCTION_INSTANTIATE variable=reuse_load_bias
  for (int i = 0; i < out_channels; i++) {
    #pragma HLS pipeline II=1 rewind
    bias[i] = bias_data[i];
  }
}