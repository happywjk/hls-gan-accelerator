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
  float input_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  int batch_size,
  int channels,
  int tile_height,
  int tile_width
) {
  // #pragma HLS INLINE OFF
  // #pragma HLS FUNCTION_INSTANTIATE variable=reuse_relu
  for (int batch = 0; batch < batch_size; batch++) {
      for (int channel = 0; channel < channels; channel++) {
          #pragma HLS pipeline II=1 rewind
          for (int h = 0; h < tile_height; h++) {
              for (int w = 0; w < tile_width; w++) {
                  // Apply ReLU: max(0, x)
                  output_tile[batch][channel][h][w] = max(0.0f, input_tile[batch][channel][h][w]);
              }
          }
      }
  }
}