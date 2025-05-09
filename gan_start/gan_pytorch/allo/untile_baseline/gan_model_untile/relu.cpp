#include "relu.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Apply ReLU to the entire input
void kernel_relu_layer(
  float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int batch_size,
  int channels,
  int height,
  int width
) {
  for (int batch = 0; batch < batch_size; batch++) {
      for (int channel = 0; channel < channels; channel++) {
          #pragma HLS pipeline II=1 rewind
          for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                  // Apply ReLU: max(0, x)
                  output[batch][channel][h][w] = max(0.0f, input[batch][channel][h][w]);
              }
          }
      }
  }
}