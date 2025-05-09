#include "tanh.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Reference to the shared buffers (defined in gan_model.cpp)
extern float shared_input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];
extern float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
extern float shared_bias[MAX_CHANNELS];
extern float shared_conv_output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];

// Apply tanh activation to the entire data
void kernel_tanh_layer(
  float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int batch_size,
  int channels,
  int height,
  int width
) {
  for (int batch = 0; batch < batch_size; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          #pragma HLS pipeline II=1 rewind
          // Apply tanh function: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
          float input_value = input[batch][channel][row][col];
          output[batch][channel][row][col] = tanh(input_value);
        }
      }
    }
  }
}