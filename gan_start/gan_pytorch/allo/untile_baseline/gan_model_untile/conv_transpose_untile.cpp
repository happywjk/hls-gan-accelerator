#include "conv_transpose.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;


// Process the entire input for transposed convolution
void kernel_convolution_transpose_layer(
  float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float bias[MAX_CHANNELS],
  float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int in_channels,
  int out_channels,
  int input_height,
  int input_width,
  int output_height,
  int output_width,
  int stride,
  int padding
) {
  // Process each output position
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int out_c = 0; out_c < out_channels; out_c++) {
      for (int out_h = 0; out_h < output_height; out_h++) {
        for (int out_w = 0; out_w < output_width; out_w++) {
          // Initialize with bias
          float sum = bias[out_c];
          
          // Perform standard convolution operation
          for (int in_c = 0; in_c < in_channels; in_c++) {
            #pragma HLS unroll factor=UNROLL_FACTOR
            for (int k_h = 0; k_h < KERNEL_SIZE; k_h++) {
              #pragma HLS unroll factor=UNROLL_FACTOR
              for (int k_w = 0; k_w < KERNEL_SIZE; k_w++) {
                #pragma HLS unroll factor=UNROLL_FACTOR               
                sum += weights[in_c][out_c][k_h][k_w] * 
                input[batch][in_c][out_h+k_h][out_w+k_w];
              
              }
            }
          }           
          // Store result
          output[batch][out_c][out_h][out_w] = sum;
        }
      }
    }
  }
}

// Load weights for processing
void load_weights(
  float *weight_data,
  float weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  int in_channels,
  int out_channels
) {
  const int weight_channel_size = KERNEL_SIZE * KERNEL_SIZE;
  const int weight_out_channel_size = out_channels  * weight_channel_size;

  for (int in_channel = 0; in_channel < in_channels; in_channel++) {
    for (int out_channel = 0; out_channel < out_channels; out_channel++) {
      for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
        for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
          #pragma HLS pipeline II=1 rewind
          int index = (in_channel * weight_out_channel_size) + 
                      (out_channel * weight_channel_size) + 
                      (kernel_row * KERNEL_SIZE) + kernel_col;
          
          // Flip the kernel for transposed convolution
          int flipped_row = KERNEL_SIZE - 1 - kernel_row;
          int flipped_col = KERNEL_SIZE - 1 - kernel_col;
          weights[in_channel][out_channel][flipped_row][flipped_col] = weight_data[index];
        }
      }
    }
  }
}

// Load bias data
void load_bias(
  float *bias_data,
  float bias[MAX_CHANNELS],
  int out_channels
) {
  for (int i = 0; i < out_channels; i++) {
    #pragma HLS pipeline II=1 rewind
    bias[i] = bias_data[i];
  }
}