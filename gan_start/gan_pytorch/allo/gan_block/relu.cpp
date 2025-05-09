#include "relu.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Apply ReLU to a single tile
void kernel_relu_layer_tile(
  float input_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1]
) {
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < OUT_CHANNELS; channel++) {
      // Get input value at current position
      float input_val = input_tile[batch][channel][0][0];
      
      // ReLU activation: max(0, input_val)
      float output_val = (input_val > 0) ? input_val : 0;
      
      // Store the result
      output_tile[batch][channel][0][0] = output_val;
    }
  }
}