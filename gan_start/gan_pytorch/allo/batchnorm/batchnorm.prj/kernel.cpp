//===------------------------------------------------------------*- C++ -*-===//
//
// Simplified BatchNorm2d implementation with parameterized tile-based processing
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Global parameters
const int BATCH_SIZE = 4;
const int NUM_CHANNELS = 16;
const int HEIGHT = 64;
const int WIDTH = 64;

extern "C" {

// Process a single tile with BatchNorm
void kernel_batchnorm_layer_tile(
  float input_tile[BATCH_SIZE][NUM_CHANNELS][1][1],
  float gamma[NUM_CHANNELS],
  float beta[NUM_CHANNELS],
  float running_mean[NUM_CHANNELS],
  float running_var[NUM_CHANNELS],
  float output_tile[BATCH_SIZE][NUM_CHANNELS][1][1]
) {
  // Process a single tile with BatchNorm
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < NUM_CHANNELS; channel++) {
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

// Load a single input tile from global memory
void load_input_tile(
  float *input_data,
  float input_tile[BATCH_SIZE][NUM_CHANNELS][1][1],
  int row,
  int col
) {
  const int channel_size = HEIGHT * WIDTH;
  const int batch_size = NUM_CHANNELS * channel_size;
  
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < NUM_CHANNELS; channel++) {
      #pragma HLS pipeline II=1 rewind
      // Check if within bounds
      if (row < HEIGHT && col < WIDTH) {
        // Calculate index in the flattened input array
        int index = batch * batch_size + channel * channel_size + row * WIDTH + col;
        input_tile[batch][channel][0][0] = input_data[index];
      } else {
        input_tile[batch][channel][0][0] = 0.0f;
      }
    }
  }
}

// Load BatchNorm parameters
void load_batchnorm_params(
  float *gamma_data,
  float *beta_data,
  float *running_mean_data,
  float *running_var_data,
  float gamma[NUM_CHANNELS],
  float beta[NUM_CHANNELS],
  float running_mean[NUM_CHANNELS],
  float running_var[NUM_CHANNELS]
) {
  for (int i = 0; i < NUM_CHANNELS; i++) {
    #pragma HLS pipeline II=1 rewind
    gamma[i] = gamma_data[i];
    beta[i] = beta_data[i];
    running_mean[i] = running_mean_data[i];
    running_var[i] = running_var_data[i];
  }
}

// Store a single output tile to global memory
void store_output_tile(
  float *output_data,
  float output_tile[BATCH_SIZE][NUM_CHANNELS][1][1],
  int row,
  int col
) {
  const int channel_size = HEIGHT * WIDTH;
  const int batch_size = NUM_CHANNELS * channel_size;
  
  // Check if the output tile position is within bounds
  if (row < HEIGHT && col < WIDTH) {
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
      for (int channel = 0; channel < NUM_CHANNELS; channel++) {
        #pragma HLS pipeline II=1 rewind
        // Calculate index in the flattened output array
        int index = batch * batch_size + channel * channel_size + row * WIDTH + col;
        output_data[index] = output_tile[batch][channel][0][0];
      }
    }
  }
}

void top(
  float *input_data,
  float *gamma_data,
  float *beta_data,
  float *running_mean_data,
  float *running_var_data,
  float *output_data
) {
  #pragma HLS interface m_axi port=input_data offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=gamma_data offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=beta_data offset=slave bundle=gmem2
  #pragma HLS interface m_axi port=running_mean_data offset=slave bundle=gmem3
  #pragma HLS interface m_axi port=running_var_data offset=slave bundle=gmem4
  #pragma HLS interface m_axi port=output_data offset=slave bundle=gmem5
  
  // Allocate on-chip buffers (BRAM)
  float input_tile[BATCH_SIZE][NUM_CHANNELS][1][1];
  float output_tile[BATCH_SIZE][NUM_CHANNELS][1][1];
  float gamma[NUM_CHANNELS];
  float beta[NUM_CHANNELS];
  float running_mean[NUM_CHANNELS];
  float running_var[NUM_CHANNELS];
  
  // Load BatchNorm parameters once (they're small)
  load_batchnorm_params(gamma_data, beta_data, running_mean_data, running_var_data,
                       gamma, beta, running_mean, running_var);
  
  // Process each tile position
  for (int row = 0; row < HEIGHT; row++) {
    for (int col = 0; col < WIDTH; col++) {
      // 1. Load input tile
      load_input_tile(input_data, input_tile, row, col);
      
      // 2. Process the tile
      kernel_batchnorm_layer_tile(input_tile, gamma, beta, running_mean, running_var, output_tile);
      
      // 3. Store output tile
      store_output_tile(output_data, output_tile, row, col);
    }
  }
}

} // extern "C"