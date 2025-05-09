#include "conv_transpose.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;



void kernel_convolution_layer_tile(
  float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float bias[OUT_CHANNELS],
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1]
) {
  // This kernel processes a tile and produces a 1x1 output tile
  
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++) {
      float sum = bias[out_channel];
      
      for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++) {
        #pragma HLS unroll factor=UNROLL_FACTOR
        for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
          #pragma HLS unroll factor=UNROLL_FACTOR
          for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
            #pragma HLS unroll factor=UNROLL_FACTOR
            sum += weights[in_channel][out_channel][kernel_row][kernel_col] * 
                   input_tile[batch][in_channel][kernel_row][kernel_col];
          }
        }
      }
      
      output_tile[batch][out_channel][0][0] = sum;
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