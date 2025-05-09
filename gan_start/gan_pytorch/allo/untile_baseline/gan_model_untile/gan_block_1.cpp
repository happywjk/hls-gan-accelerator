#include "gan_block_1.h"
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
#include <stdio.h>
using namespace std;


extern "C" {
  void top_block1(
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
    
    // Print first few input values
    // printf("DEBUG - First 5 input values:\n");
    // for (int i = 0; i < 1 && i < gan_block1::IN_CHANNELS; i++) {
    //   printf("[%d]: %.6f\n", i, input_data[i]);
    // }
    
    // Load all parameters once
    // printf("DEBUG - Before loading weights\n");
    load_weights(weight_data, shared_weights, gan_block1::IN_CHANNELS, gan_block1::OUT_CHANNELS);
    // printf("DEBUG - After loading weights\n");
    
    // printf("DEBUG - Before loading bias\n");
    load_bias(bias_data, shared_bias, gan_block1::OUT_CHANNELS);
    // printf("DEBUG - After loading bias\n");
    
    // printf("DEBUG - Before loading batchnorm params\n");
    load_batchnorm_params(gamma_data, beta_data, running_mean_data, running_var_data,
                          shared_gamma, shared_beta, shared_running_mean, shared_running_var,
                          gan_block1::OUT_CHANNELS);
    // printf("DEBUG - After loading batchnorm params\n");
    
    // Print first few loaded parameters for verification
    // printf("DEBUG - First few loaded weights:\n");
    // for (int h = 0; h < 4; h++) {
    //   for (int w = 0; w < 4; w++) {
    //     printf("weight[0][0][%d][%d]: %.6f\n", h, w, shared_weights[0][0][h][w]);
    //   }
    // }
    
    // printf("DEBUG - First few loaded biases:\n");
    // for (int i = 0; i < 1; i++) {
    //   printf("bias[%d]: %.6f\n", i, shared_bias[i]);
    // }
    
    // printf("DEBUG - First few loaded batchnorm params:\n");
    // for (int i = 0; i < 3 && i < gan_block1::OUT_CHANNELS; i++) {
    //   printf("gamma[%d]: %.6f, beta[%d]: %.6f\n", i, shared_gamma[i], i, shared_beta[i]);
    //   printf("running_mean[%d]: %.6f, running_var[%d]: %.6f\n", 
    //          i, shared_running_mean[i], i, shared_running_var[i]);
    // }
    
    // 1. Load the entire input
    // printf("DEBUG - Before load_input\n");
    load_input(input_data, shared_input, 
               gan_block1::IN_CHANNELS, 
               gan_block1::INPUT_HEIGHT, 
               gan_block1::INPUT_WIDTH,
               gan_block1::OUTPUT_HEIGHT,
               gan_block1::OUTPUT_WIDTH,
               gan_block1::OUTPUT_PADDING,
               gan_block1::PADDING, 
               gan_block1::STRIDE);
    // printf("DEBUG - After load_input\n");
    
    // Print first few loaded input values after transformation
    // printf("DEBUG - First few loaded input values after transformation:\n");
    // for (int c = 0; c < 1; c++) {
    //   for (int h = 0; h < 7; h++) {
    //     for (int w = 0; w < 7; w++) {
    //       printf("input[0][%d][%d][%d]: %.6f\n", c, h, w, shared_input[0][c][h][w]);
    //     }
    //   }
    // }
    
    // 2. Process the entire input through ConvTranspose2d
    // printf("DEBUG - Before ConvTranspose2d\n");
    kernel_convolution_transpose_layer(
        shared_input, shared_weights, shared_bias, shared_conv_output,
        gan_block1::IN_CHANNELS, 
        gan_block1::OUT_CHANNELS,
        gan_block1::INPUT_HEIGHT,
        gan_block1::INPUT_WIDTH,
        gan_block1::OUTPUT_HEIGHT,
        gan_block1::OUTPUT_WIDTH,
        gan_block1::STRIDE,
        gan_block1::PADDING);
    // printf("DEBUG - After ConvTranspose2d\n");
    
    // Print first few convolution output values
    // printf("DEBUG - First few convolution output values:\n");
    // for (int c = 0; c < 1 && c < gan_block1::OUT_CHANNELS; c++) {
    //   for (int h = 0; h < 4 && h < gan_block1::OUTPUT_HEIGHT; h++) {
    //     for (int w = 0; w < 4 && w < gan_block1::OUTPUT_WIDTH; w++) {
    //       printf("conv_output[0][%d][%d][%d]: %.6f\n", c, h, w, shared_conv_output[0][c][h][w]);
    //     }
    //   }
    // }
    
    // 3. Process the entire output through BatchNorm2d
    // printf("DEBUG - Before BatchNorm2d\n");
    kernel_batchnorm_layer(
        shared_conv_output, shared_gamma, shared_beta, 
        shared_running_mean, shared_running_var, shared_bn_output,
        BATCH_SIZE, 
        gan_block1::OUT_CHANNELS, 
        gan_block1::OUTPUT_HEIGHT, 
        gan_block1::OUTPUT_WIDTH);
    // printf("DEBUG - After BatchNorm2d\n");
    
    // Print first few batchnorm output values
    // printf("DEBUG - First few batchnorm output values:\n");
    // for (int c = 0; c < 2 && c < gan_block1::OUT_CHANNELS; c++) {
    //   for (int h = 0; h < 2 && h < gan_block1::OUTPUT_HEIGHT; h++) {
    //     for (int w = 0; w < 2 && w < gan_block1::OUTPUT_WIDTH; w++) {
    //       printf("bn_output[0][%d][%d][%d]: %.6f\n", c, h, w, shared_bn_output[0][c][h][w]);
    //     }
    //   }
    // }
    
    // 4. Process the entire output through ReLU
    // printf("DEBUG - Before ReLU\n");
    kernel_relu_layer(
        shared_bn_output, shared_conv_output,
        BATCH_SIZE, 
        gan_block1::OUT_CHANNELS,
        gan_block1::OUTPUT_HEIGHT, 
        gan_block1::OUTPUT_WIDTH);
    // printf("DEBUG - After ReLU\n");
    
    // Print first few ReLU output values
    // printf("DEBUG - First few ReLU output values:\n");
    // for (int c = 0; c < 2 && c < gan_block1::OUT_CHANNELS; c++) {
    //   for (int h = 0; h < 2 && h < gan_block1::OUTPUT_HEIGHT; h++) {
    //     for (int w = 0; w < 2 && w < gan_block1::OUTPUT_WIDTH; w++) {
    //       printf("relu_output[0][%d][%d][%d]: %.6f\n", c, h, w, shared_conv_output[0][c][h][w]);
    //     }
    //   }
    // }
    
    // 5. Store the entire output
    // printf("DEBUG - Before store_output\n");
    store_output(
        output_data, shared_conv_output,
        BATCH_SIZE, 
        gan_block1::OUT_CHANNELS,
        gan_block1::OUTPUT_HEIGHT, 
        gan_block1::OUTPUT_WIDTH);
    // printf("DEBUG - After store_output\n");
    
    // Print first few final output values
    // printf("DEBUG - First few final output values:\n");
    // for (int i = 0; i < 10 && i < (gan_block1::OUT_CHANNELS * gan_block1::OUTPUT_HEIGHT * gan_block1::OUTPUT_WIDTH); i++) {
    //   printf("output_data[%d]: %.6f\n", i, output_data[i]);
    // }
  }
}