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
  float input_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
  float gamma[OUT_CHANNELS],
  float beta[OUT_CHANNELS],
  float running_mean[OUT_CHANNELS],
  float running_var[OUT_CHANNELS],
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1]
) {
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < OUT_CHANNELS; channel++) {
      // Precompute channel-specific constants
      float mean_c = running_mean[channel];
      float var_c = running_var[channel];
      float gamma_c = gamma[channel];
      float beta_c = beta[channel];
      
      // Add epsilon for numerical stability
      float var_eps = var_c + 0.00001f;
      
      // Calculate standard deviation
      float std_dev = sqrt(var_eps);
      
      // Precompute scale and shift for efficiency
      float scale = gamma_c / std_dev;
      float shift = beta_c - (mean_c * scale);
      
      // Apply the BatchNorm operation directly to the single element
      float input_val = input_tile[batch][channel][0][0]; 
      output_tile[batch][channel][0][0] = input_val * scale + shift;
    }
  }
}
  
// Load batch normalization parameters
void load_batchnorm_params(
  float *gamma_data,
  float *beta_data,
  float *running_mean_data,
  float *running_var_data,
  float gamma[OUT_CHANNELS],
  float beta[OUT_CHANNELS],
  float running_mean[OUT_CHANNELS],
  float running_var[OUT_CHANNELS]
) {
  for (int i = 0; i < OUT_CHANNELS; i++) {
    #pragma HLS pipeline II=1 rewind
    gamma[i] = gamma_data[i];
    beta[i] = beta_data[i];
    running_mean[i] = running_mean_data[i];
    running_var[i] = running_var_data[i];
  }
}