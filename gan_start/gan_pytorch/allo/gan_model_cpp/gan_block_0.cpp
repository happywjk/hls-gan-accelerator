#include "gan_block_0.h"
#include "gan_block_base.h"
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

  // static void load_input_tile(
  //   float *input_data,
  //   float input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  //   int row_start,
  //   int col_start,
  //   int in_channels,
  //   int input_height,
  //   int input_width,
  //   int padding,
  //   int stride
  // ) {
  //   const int input_channel_size = input_height * input_width;
  //   const int input_batch_size = in_channels * input_channel_size;
  
  //   for (int batch = 0; batch < BATCH_SIZE; batch++) {
  //     for (int channel = 0; channel < in_channels; channel++) {
  //       for (int row = 0; row < KERNEL_SIZE; row++) {
  //         for (int col = 0; col < KERNEL_SIZE; col++) {
  //           #pragma HLS pipeline II=1 rewind
            
  //           // Calculate actual input position with padding adjustment
  //           int in_row_no_stride = row_start + row - KERNEL_SIZE + 1 + padding;
  //           int in_col_no_stride = col_start + col - KERNEL_SIZE + 1 + padding;
  //           int in_row = (in_row_no_stride) / stride;
  //           int in_col = (in_col_no_stride) / stride;
            
  //           bool is_orig_input_pos = (in_row_no_stride % stride == 0) && (in_col_no_stride % stride == 0);
  //           // Check if within input bounds
  //           if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width && is_orig_input_pos) {
  //             // Calculate index in the flattened input array
  //             int index = (batch * input_batch_size) + 
  //                         (channel * input_channel_size) + 
  //                         (in_row * input_width) + in_col;
  //             input_tile[batch][channel][row][col] = input_data[index];
  //           } else {
  //             // Zero padding for out-of-bounds
  //             input_tile[batch][channel][row][col] = 0.0f;
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  
  void top_block0(
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
    
    // Load all parameters once (assuming they fit in BRAM)
    load_weights_tile(weight_data, shared_weights, gan_block0::IN_CHANNELS, gan_block0::OUT_CHANNELS);
    load_bias_tile(bias_data, shared_bias, gan_block0::OUT_CHANNELS);
    load_batchnorm_params(gamma_data, beta_data, running_mean_data, running_var_data,
                          shared_gamma, shared_beta, shared_running_mean, shared_running_var,
                          gan_block0::OUT_CHANNELS);
    
    // Process each output position
    for (int out_row = 0; out_row < gan_block0::OUTPUT_HEIGHT; out_row++) { 
      for (int out_col = 0; out_col < gan_block0::OUTPUT_WIDTH; out_col++) {
        // 1. Load the input tile for this position
        load_input_tile(input_data, shared_input_tile, out_row, out_col, 
          gan_block0::IN_CHANNELS, gan_block0::INPUT_HEIGHT, gan_block0::INPUT_WIDTH,
          gan_block0::PADDING, gan_block0::STRIDE);
        
        // 2. Process the tile through ConvTranspose2d
        kernel_convolution_layer_tile(shared_input_tile, shared_weights, shared_bias, shared_conv_output_tile,
                                     gan_block0::IN_CHANNELS, gan_block0::OUT_CHANNELS);
        
        // 3. Process the tile through BatchNorm2d
        kernel_batchnorm_layer_tile(shared_conv_output_tile, shared_gamma, shared_beta, 
                                   shared_running_mean, shared_running_var, shared_bn_output_tile,
                                   BATCH_SIZE, gan_block0::OUT_CHANNELS, 
                                   1, 1);  // Assuming 1x1 output tile size
        
        // 4. Process the tile through ReLU
        kernel_relu_layer_tile(shared_bn_output_tile, shared_conv_output_tile,
          BATCH_SIZE, gan_block0::OUT_CHANNELS,
          1, 1);  // Assuming 1x1 output tile size // Assuming 1x1 output tile size
        
        // 5. Store the final output tile
        store_output_tile(output_data, shared_conv_output_tile, out_row, out_col,
                         BATCH_SIZE, gan_block0::OUT_CHANNELS,
                         gan_block0::OUTPUT_HEIGHT, gan_block0::OUTPUT_WIDTH);
      }
    }
  }
}