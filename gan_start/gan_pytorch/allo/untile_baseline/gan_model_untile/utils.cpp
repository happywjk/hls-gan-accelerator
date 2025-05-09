#include "utils.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Load the entire input
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
) {
  const int input_channel_size = input_height * input_width;
  const int input_batch_size = in_channels * input_channel_size;

  // Initialize the entire input array with zeros (for padding)
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < in_channels; channel++) {
      for (int h = 0; h < output_padding; h++) {
        for (int w = 0; w < output_padding; w++) {
          input[batch][channel][h][w] = 0.0f;
        }
      }
    }
  }

  // Load actual input data with appropriate striding and padding
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < in_channels; channel++) {
      for (int in_h = 0; in_h < input_height; in_h++) {
        for (int in_w = 0; in_w < input_width; in_w++) {
          #pragma HLS pipeline II=1 rewind
          
          // Calculate index in the flattened input array
          int index = (batch * input_batch_size) + 
                      (channel * input_channel_size) + 
                      (in_h * input_width) + in_w;
          
          // Calculate the corresponding position in the output array for transposed convolution
          int out_h = in_h * stride + KERNEL_SIZE - 1 - padding;
          int out_w = in_w * stride + KERNEL_SIZE - 1 - padding;
          
          // Place the input data at the calculated position
          input[batch][channel][out_h][out_w] = input_data[index];
        }
      }
    }
  }
}
// Store the entire output to global memory
void store_output(
  float *output_data,
  float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int batch_size,
  int out_channels,
  int output_height,
  int output_width
) {
  const int output_channel_size = output_height * output_width;
  const int output_batch_size = out_channels * output_channel_size;

  for (int batch = 0; batch < batch_size; batch++) {
    for (int channel = 0; channel < out_channels; channel++) {
      for (int h = 0; h < output_height; h++) {
        for (int w = 0; w < output_width; w++) {
          #pragma HLS pipeline II=1 rewind
          
          // Calculate index in the flattened output array
          int index = (batch * output_batch_size) + 
                      (channel * output_channel_size) + 
                      (h * output_width) + w;
          
          output_data[index] = output[batch][channel][h][w];
        }
      }
    }
  }
}