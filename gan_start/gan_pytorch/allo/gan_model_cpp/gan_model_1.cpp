#include "gan_block_0.h"
#include "gan_block_1.h"
#include "gan_block_base.h"
#include "conv_transpose.h"
#include "batchnorm.h"
#include "relu.h"
#include "utils.h"
#include "gan_model_1.h"
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

// Shared buffers for all GAN blocks
// Using maximum dimensions to accommodate all layers
extern "C" {
// Input and parameter buffers
float shared_input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
float shared_bias[MAX_CHANNELS];
float shared_gamma[MAX_CHANNELS];
float shared_beta[MAX_CHANNELS];
float shared_running_mean[MAX_CHANNELS];
float shared_running_var[MAX_CHANNELS];

// Intermediate output buffers
float shared_conv_output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE];
float shared_bn_output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE];

// Top-level GAN generator function
void top(
    // Input noise vector
    float *noise_input,
    
    // Block 0 parameters (128 -> 1024, 1x1 -> 4x4)
    float *weights_0,
    float *bias_0,
    float *gamma_0,
    float *beta_0,
    float *running_mean_0,
    float *running_var_0,
    
    // Block 1 parameters (1024 -> 512, 4x4 -> 8x8)
    float *weights_1,
    float *bias_1,
    float *gamma_1,
    float *beta_1,
    float *running_mean_1,
    float *running_var_1,
    
    // Final generated image output
    float *output_image)
{
    #pragma HLS INTERFACE m_axi port=noise_input offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=weights_0 offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=bias_0 offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=gamma_0 offset=slave bundle=gmem3
    #pragma HLS INTERFACE m_axi port=beta_0 offset=slave bundle=gmem4
    #pragma HLS INTERFACE m_axi port=running_mean_0 offset=slave bundle=gmem5
    #pragma HLS INTERFACE m_axi port=running_var_0 offset=slave bundle=gmem6
    
    #pragma HLS INTERFACE m_axi port=weights_1 offset=slave bundle=gmem7
    #pragma HLS INTERFACE m_axi port=bias_1 offset=slave bundle=gmem8
    #pragma HLS INTERFACE m_axi port=gamma_1 offset=slave bundle=gmem9
    #pragma HLS INTERFACE m_axi port=beta_1 offset=slave bundle=gmem10
    #pragma HLS INTERFACE m_axi port=running_mean_1 offset=slave bundle=gmem11
    #pragma HLS INTERFACE m_axi port=running_var_1 offset=slave bundle=gmem12
    
    
    #pragma HLS INTERFACE m_axi port=output_image offset=slave bundle=gmem33
    
    #pragma HLS INTERFACE s_axilite port=return 
    
    // Allocate memory for intermediate outputs between blocks
    float *block0_output = (float *)malloc(BATCH_SIZE * 1024 * 4 * 4 * sizeof(float));

    // Block 0: 128 -> 1024, 1x1 -> 4x4
    top_block0(
        noise_input,     // Input: noise vector
        weights_0,       // Weights
        bias_0,          // Bias
        gamma_0,         // BN gamma
        beta_0,          // BN beta
        running_mean_0,  // BN running mean
        running_var_0,   // BN running var
        block0_output    // Output: 1024 channels, 4x4 size
    );
    
    // Block 1: 1024 -> 512, 4x4 -> 8x8
    top_block1(
        block0_output,   // Input from block 0
        weights_1,       // Weights
        bias_1,          // Bias
        gamma_1,         // BN gamma
        beta_1,          // BN beta
        running_mean_1,  // BN running mean
        running_var_1,   // BN running var
        output_image    // Output: 512 channels, 8x8 size
    );
    
    // Block 2: 512 -> 256, 8x8 -> 16x16

    // Free allocated memory
    free(block0_output);
}
}