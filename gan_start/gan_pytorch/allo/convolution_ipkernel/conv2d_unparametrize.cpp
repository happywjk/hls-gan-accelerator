#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

extern "C" {

// Process a single 4x4 tile of the input
void kernel_convolution_layer_tile(
  float input_tile[4][3][4][4],
  float weights[16][3][4][4],
  float bias[16],
  float output_tile[4][16][1][1]
) {
  // This kernel processes a 4x4 input tile and produces a 1x1 output tile
  
  for (int batch = 0; batch < 4; batch++) {
    for (int out_channel = 0; out_channel < 16; out_channel++) {
      float sum = bias[out_channel];
      
      for (int in_channel = 0; in_channel < 3; in_channel++) {
        #pragma HLS unroll factor=1
        for (int kernel_row = 0; kernel_row < 4; kernel_row++) {
          #pragma HLS unroll factor=1
          for (int kernel_col = 0; kernel_col < 4; kernel_col++) {
            #pragma HLS unroll factor=1
            sum += weights[out_channel][in_channel][kernel_row][kernel_col] * 
                   input_tile[batch][in_channel][kernel_row][kernel_col];
          }
        }
      }
      
      output_tile[batch][out_channel][0][0] = sum;
    }
  }
}

// Load a single input tile from global memory with proper padding
void load_input_tile(
  float *input_data,
  float input_tile[4][3][4][4],
  int row_start,
  int col_start
) {
  for (int batch = 0; batch < 4; batch++) {
    for (int channel = 0; channel < 3; channel++) {
      for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
          #pragma HLS pipeline II=1 rewind
          
          // Calculate actual input position with padding adjustment
          // Subtract 1 to account for the 1-pixel padding in original implementation
          int in_row = row_start + row - 1;
          int in_col = col_start + col - 1;
          
          // Check if within input bounds (64x64)
          if (in_row >= 0 && in_row < 64 && in_col >= 0 && in_col < 64) {
            // Calculate index in the flattened input array
            int index = (batch * 12288) + (channel * 4096) + 
                       (in_row * 64) + in_col;
            input_tile[batch][channel][row][col] = input_data[index];
          } else {
            // Zero padding for out-of-bounds (both at beginning and end)
            input_tile[batch][channel][row][col] = 0.0f;
          }
        }
      }
    }
  }
}

// Load weights for processing
void load_weights_tile(
  float *weight_data,
  float weights[16][3][4][4]
) {
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 4; l++) {
          #pragma HLS pipeline II=1 rewind
          weights[i][j][k][l] = weight_data[(i * 48) + (j * 16) + (k * 4) + l];
        }
      }
    }
  }
}

// Load bias data
void load_bias_tile(
  float *bias_data,
  float bias[16]
) {
  for (int i = 0; i < 16; i++) {
    #pragma HLS pipeline II=1 rewind
    bias[i] = bias_data[i];
  }
}

// Store a single output tile to global memory
void store_output_tile(
  float *output_data,
  float output_tile[4][16][1][1],
  int row,
  int col
) {
  // Check if the output tile position is within bounds
  if (row < 63 && col < 63) {
    for (int batch = 0; batch < 4; batch++) {
      for (int channel = 0; channel < 16; channel++) {
        #pragma HLS pipeline II=1 rewind
        
        // Calculate index in the flattened output array
        int index = (batch * 63504) + (channel * 3969) + (row * 63) + col;
        output_data[index] = output_tile[batch][channel][0][0];
      }
    }
  }
}

void top(
  float *input_data,
  float *weight_data,
  float *bias_data,
  float *output_data
) {
  #pragma HLS interface m_axi port=input_data offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=weight_data offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=bias_data offset=slave bundle=gmem2
  #pragma HLS interface m_axi port=output_data offset=slave bundle=gmem3
  
  // Allocate on-chip buffers (BRAM)
  float input_tile[4][3][4][4];
  float weights[16][3][4][4];
  float bias[16];
  float output_tile[4][16][1][1];
  
  // Load weights and biases once (assuming they fit in BRAM)
  load_weights_tile(weight_data, weights);
  load_bias_tile(bias_data, bias);
  
  // Process each output position in the 63x63 output
  for (int out_row = 0; out_row < 63; out_row++) {
    for (int out_col = 0; out_col < 63; out_col++) {
      // Calculate corresponding input position
      // Adding 1 to account for the padding in the original implementation
      int in_row = out_row + 1;
      int in_col = out_col + 1;
      
      // 1. Load the input tile for this position
      load_input_tile(input_data, input_tile, in_row, in_col);
      
      // 2. Process the tile - launch kernel
      kernel_convolution_layer_tile(input_tile, weights, bias, output_tile);
      
      // 3. Store the output tile
      store_output_tile(output_data, output_tile, out_row, out_col);
    }
  }
}

}