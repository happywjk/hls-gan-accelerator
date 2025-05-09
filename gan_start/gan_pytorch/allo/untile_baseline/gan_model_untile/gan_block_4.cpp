#include "gan_block_4.h"
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

// External buffer declarations with correct dimensions for untiled approach
extern float shared_input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];
extern float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
extern float shared_bias[MAX_CHANNELS];
extern float shared_gamma[MAX_CHANNELS];
extern float shared_beta[MAX_CHANNELS];
extern float shared_running_mean[MAX_CHANNELS];
extern float shared_running_var[MAX_CHANNELS];
extern float shared_conv_output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];
extern float shared_bn_output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];

extern "C" {
  void top_block4(
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
    
    // Load all parameters once
    load_weights(weight_data, shared_weights, gan_block4::IN_CHANNELS, gan_block4::OUT_CHANNELS);
    load_bias(bias_data, shared_bias, gan_block4::OUT_CHANNELS);
    load_batchnorm_params(gamma_data, beta_data, running_mean_data, running_var_data,
                          shared_gamma, shared_beta, shared_running_mean, shared_running_var,
                          gan_block4::OUT_CHANNELS);
    
    // 1. Load the entire input
    load_input(input_data, shared_input, 
               gan_block4::IN_CHANNELS, 
               gan_block4::INPUT_HEIGHT, 
               gan_block4::INPUT_WIDTH,
               gan_block4::OUTPUT_HEIGHT,
               gan_block4::OUTPUT_WIDTH,
               gan_block4::OUTPUT_PADDING,
               gan_block4::PADDING, 
               gan_block4::STRIDE);
    
    // 2. Process the entire input through ConvTranspose2d
    kernel_convolution_transpose_layer(
        shared_input, shared_weights, shared_bias, shared_conv_output,
        gan_block4::IN_CHANNELS, 
        gan_block4::OUT_CHANNELS,
        gan_block4::INPUT_HEIGHT,
        gan_block4::INPUT_WIDTH,
        gan_block4::OUTPUT_HEIGHT,
        gan_block4::OUTPUT_WIDTH,
        gan_block4::STRIDE,
        gan_block4::PADDING);
    
    // 3. Process the entire output through BatchNorm2d
    kernel_batchnorm_layer(
        shared_conv_output, shared_gamma, shared_beta, 
        shared_running_mean, shared_running_var, shared_bn_output,
        BATCH_SIZE, 
        gan_block4::OUT_CHANNELS, 
        gan_block4::OUTPUT_HEIGHT, 
        gan_block4::OUTPUT_WIDTH);
    
    // 4. Process the entire output through ReLU
    kernel_relu_layer(
        shared_bn_output, shared_conv_output,
        BATCH_SIZE, 
        gan_block4::OUT_CHANNELS,
        gan_block4::OUTPUT_HEIGHT, 
        gan_block4::OUTPUT_WIDTH);
    
    // 5. Store the entire output
    store_output(
        output_data, shared_conv_output,
        BATCH_SIZE, 
        gan_block4::OUT_CHANNELS,
        gan_block4::OUTPUT_HEIGHT, 
        gan_block4::OUTPUT_WIDTH);
  }
}