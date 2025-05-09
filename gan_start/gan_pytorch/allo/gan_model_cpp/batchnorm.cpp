#include "batchnorm.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

void kernel_batchnorm_layer_tile(
  float input_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  float gamma[MAX_CHANNELS],
  float beta[MAX_CHANNELS],
  float running_mean[MAX_CHANNELS],
  float running_var[MAX_CHANNELS],
  float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
  int batch_size,
  int channels,
  int tile_height,
  int tile_width
) {
  // #pragma HLS INLINE OFF
  // #pragma HLS FUNCTION_INSTANTIATE variable=reuse_batchnorm
  const float epsilon = 1e-5f;
  
  for (int batch = 0; batch < batch_size; batch++) {
      for (int channel = 0; channel < channels; channel++) {
          #pragma HLS pipeline II=1 rewind
          for (int h = 0; h < tile_height; h++) {
              for (int w = 0; w < tile_width; w++) {
                  // Apply BatchNorm formula: y = gamma * (x - mean) / sqrt(var + epsilon) + beta
                  float normalized = (input_tile[batch][channel][h][w] - running_mean[channel]) / 
                                    sqrt(running_var[channel] + epsilon);
                  output_tile[batch][channel][h][w] = gamma[channel] * normalized + beta[channel];
              }
          }
      }
  }
}

// Load batchnorm parameters implementation
void load_batchnorm_params(
  float *gamma_data,
  float *beta_data,
  float *running_mean_data,
  float *running_var_data,
  float gamma[MAX_CHANNELS],
  float beta[MAX_CHANNELS],
  float running_mean[MAX_CHANNELS],
  float running_var[MAX_CHANNELS],
  int channels
) {
  // #pragma HLS INLINE OFF
  // #pragma HLS FUNCTION_INSTANTIATE variable=reuse_batchnormparams
  for (int i = 0; i < channels; i++) {
      #pragma HLS pipeline II=1 rewind
      gamma[i] = gamma_data[i];
      beta[i] = beta_data[i];
      running_mean[i] = running_mean_data[i];
      running_var[i] = running_var_data[i];
  }
}