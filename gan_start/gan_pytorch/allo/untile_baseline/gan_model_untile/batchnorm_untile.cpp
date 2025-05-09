#include "batchnorm.h"
#include <algorithm>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
using namespace std;

void kernel_batchnorm_layer(
  float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  float gamma[MAX_CHANNELS],
  float beta[MAX_CHANNELS],
  float running_mean[MAX_CHANNELS],
  float running_var[MAX_CHANNELS],
  float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
  int batch_size,
  int channels,
  int height,
  int width
) {
  const float epsilon = 1e-5f;
  
  // Print BatchNorm parameters for verification
  // printf("DEBUG - BatchNorm Parameters:\n");
  // for (int i = 0; i < min(5, channels); i++) {
  //   printf("Channel %d: gamma=%.6f, beta=%.6f, mean=%.6f, var=%.6f\n", 
  //          i, gamma[i], beta[i], running_mean[i], running_var[i]);
  // }
  
  for (int batch = 0; batch < batch_size; batch++) {
    for (int channel = 0; channel < channels; channel++) {
      #pragma HLS pipeline II=1 rewind
      
      // Pre-compute the normalization factor for this channel
      float norm_factor = 1.0f / sqrt(running_var[channel] + epsilon);
      float channel_mean = running_mean[channel];
      float channel_gamma = gamma[channel];
      float channel_beta = beta[channel];
      
      // Debug info for first few elements
      // if (channel < 2) {
      //   printf("Channel %d: norm_factor=%.6f\n", channel, norm_factor);
      // }
      
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          // Apply BatchNorm formula: y = gamma * (x - mean) / sqrt(var + epsilon) + beta
          float normalized = (input[batch][channel][h][w] - channel_mean) * norm_factor;
          output[batch][channel][h][w] = channel_gamma * normalized + channel_beta;
          
          // Debug detailed calculation for first few elements
          // if (batch == 0 && channel < 2 && h < 2 && w < 2) {
          //   printf("DEBUG - BatchNorm Calculation [%d][%d][%d][%d]:\n", batch, channel, h, w);
          //   printf("  Input: %.6f\n", input[batch][channel][h][w]);
          //   printf("  Mean: %.6f\n", channel_mean);
          //   printf("  Step1 (x-mean): %.6f\n", input[batch][channel][h][w] - channel_mean);
          //   printf("  Step2 ((x-mean)/sqrt(var+eps)): %.6f\n", normalized);
          //   printf("  Step3 (gamma*norm+beta): %.6f\n", output[batch][channel][h][w]);
          // }
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
  for (int i = 0; i < channels; i++) {
    #pragma HLS pipeline II=1 rewind
    gamma[i] = gamma_data[i];
    beta[i] = beta_data[i];
    running_mean[i] = running_mean_data[i];
    running_var[i] = running_var_data[i];
  }
}