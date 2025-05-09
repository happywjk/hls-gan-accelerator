#include "gan_block.h"
#include "conv_transpose.h"
#include "batchnorm.h"
#include "relu.h"
#include "utils.h"
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

  void load_input_tile(
    float *input_data,
    float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    int row_start,
    int col_start
  ) {
    const int input_channel_size = INPUT_HEIGHT * INPUT_WIDTH;
    const int input_batch_size = IN_CHANNELS * input_channel_size;
  
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
      for (int channel = 0; channel < IN_CHANNELS; channel++) {
        for (int row = 0; row < KERNEL_SIZE; row++) {
          for (int col = 0; col < KERNEL_SIZE; col++) {
            #pragma HLS pipeline II=1 rewind
            
            // Calculate actual input position with padding adjustment
            int in_row_no_stride = row_start + row - KERNEL_SIZE + 1 + PADDING;
            int in_col_no_stride = col_start + col - KERNEL_SIZE + 1 + PADDING;
            int in_row = (in_row_no_stride) / stride;
            int in_col = (in_col_no_stride) / stride;
            
            bool is_orig_input_pos = (in_row_no_stride % stride == 0) && (in_col_no_stride % stride == 0);
            // Check if within input bounds
            if (in_row >= 0 && in_row < INPUT_HEIGHT && in_col >= 0 && in_col < INPUT_WIDTH && is_orig_input_pos) {
              // Calculate index in the flattened input array
              int index = (batch * input_batch_size) + 
                          (channel * input_channel_size) + 
                          (in_row * INPUT_WIDTH) + in_col;
              input_tile[batch][channel][row][col] = input_data[index];
            } else {
              // Zero padding for out-of-bounds
              input_tile[batch][channel][row][col] = 0.0f;
            }
          }
        }
      }
    }
  }
void top(
    float *input_data,
    float *weight_data,
    float *bias_data,
    float *gamma_data,
    float *beta_data,
    float *running_mean_data,
    float *running_var_data,
    float *output_data
  ) {
    #pragma HLS interface m_axi port=input_data offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=weight_data offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=bias_data offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=gamma_data offset=slave bundle=gmem3
    #pragma HLS interface m_axi port=beta_data offset=slave bundle=gmem4
    #pragma HLS interface m_axi port=running_mean_data offset=slave bundle=gmem5
    #pragma HLS interface m_axi port=running_var_data offset=slave bundle=gmem6
    #pragma HLS interface m_axi port=output_data offset=slave bundle=gmem7
    
    // Allocate on-chip buffers (BRAM)
    float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    float bias[OUT_CHANNELS];
    float gamma[OUT_CHANNELS];
    float beta[OUT_CHANNELS];
    float running_mean[OUT_CHANNELS];
    float running_var[OUT_CHANNELS];
    
    // Intermediate buffers for layer outputs
    float conv_output_tile[BATCH_SIZE][OUT_CHANNELS][1][1];
    float bn_output_tile[BATCH_SIZE][OUT_CHANNELS][1][1];
    
    // Load all parameters once (assuming they fit in BRAM)
    load_weights_tile(weight_data, weights);
    load_bias_tile(bias_data, bias);
    load_batchnorm_params(gamma_data, beta_data, running_mean_data, running_var_data,
                          gamma, beta, running_mean, running_var);
    
    // Process each output position
    for (int out_row = 0; out_row < OUTPUT_HEIGHT; out_row++) { 
      for (int out_col = 0; out_col < OUTPUT_WIDTH; out_col++) {
        // 1. Load the input tile for this position
        load_input_tile(input_data, input_tile, out_row, out_col);
        
        // 2. Process the tile through ConvTranspose2d
        kernel_convolution_layer_tile(input_tile, weights, bias, conv_output_tile);
        
        // 3. Process the tile through BatchNorm2d
        kernel_batchnorm_layer_tile(conv_output_tile, gamma, beta, 
                                   running_mean, running_var, bn_output_tile);
        
        // 4. Process the tile through ReLU
        kernel_relu_layer_tile(bn_output_tile, conv_output_tile);
        
        // 5. Store the final output tile
        store_output_tile(output_data, conv_output_tile, out_row, out_col);
      }
    }
  }
}