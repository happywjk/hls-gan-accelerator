#include "gan_block_0.h"
#include "gan_block_1.h"
#include "gan_block_2.h"
#include "gan_block_3.h"
#include "gan_block_4.h"
#include "gan_block_5.h"
#include "gan_block_base.h"
#include "gan_model.h"
#include "conv_transpose.h"
#include "batchnorm.h"
#include "relu.h"
#include "tanh.h"
#include "utils.h"
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
    
    // Block 2 parameters (512 -> 256, 8x8 -> 16x16)
    float *weights_2,
    float *bias_2,
    float *gamma_2,
    float *beta_2,
    float *running_mean_2,
    float *running_var_2,
    
    // Block 3 parameters (256 -> 128, 16x16 -> 32x32)
    float *weights_3,
    float *bias_3,
    float *gamma_3,
    float *beta_3,
    float *running_mean_3,
    float *running_var_3,
    
    // Block 4 parameters (128 -> 64, 32x32 -> 64x64)
    float *weights_4,
    float *bias_4,
    float *gamma_4,
    float *beta_4,
    float *running_mean_4,
    float *running_var_4,
    
    // Block 5 parameters (64 -> 3, 64x64 -> 128x128)
    float *weights_5,
    float *bias_5,
    
    // Pre-allocated intermediate memory buffers
    float *block0_output,
    float *block1_output,
    float *block2_output,
    float *block3_output,
    float *block4_output,
    
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
    
    #pragma HLS INTERFACE m_axi port=weights_2 offset=slave bundle=gmem13
    #pragma HLS INTERFACE m_axi port=bias_2 offset=slave bundle=gmem14
    #pragma HLS INTERFACE m_axi port=gamma_2 offset=slave bundle=gmem15
    #pragma HLS INTERFACE m_axi port=beta_2 offset=slave bundle=gmem16
    #pragma HLS INTERFACE m_axi port=running_mean_2 offset=slave bundle=gmem17
    #pragma HLS INTERFACE m_axi port=running_var_2 offset=slave bundle=gmem18
    
    #pragma HLS INTERFACE m_axi port=weights_3 offset=slave bundle=gmem19
    #pragma HLS INTERFACE m_axi port=bias_3 offset=slave bundle=gmem20
    #pragma HLS INTERFACE m_axi port=gamma_3 offset=slave bundle=gmem21
    #pragma HLS INTERFACE m_axi port=beta_3 offset=slave bundle=gmem22
    #pragma HLS INTERFACE m_axi port=running_mean_3 offset=slave bundle=gmem23
    #pragma HLS INTERFACE m_axi port=running_var_3 offset=slave bundle=gmem24
    
    #pragma HLS INTERFACE m_axi port=weights_4 offset=slave bundle=gmem25
    #pragma HLS INTERFACE m_axi port=bias_4 offset=slave bundle=gmem26
    #pragma HLS INTERFACE m_axi port=gamma_4 offset=slave bundle=gmem27
    #pragma HLS INTERFACE m_axi port=beta_4 offset=slave bundle=gmem28
    #pragma HLS INTERFACE m_axi port=running_mean_4 offset=slave bundle=gmem29
    #pragma HLS INTERFACE m_axi port=running_var_4 offset=slave bundle=gmem30
    
    #pragma HLS INTERFACE m_axi port=weights_5 offset=slave bundle=gmem31
    #pragma HLS INTERFACE m_axi port=bias_5 offset=slave bundle=gmem32
    
    #pragma HLS INTERFACE m_axi port=block0_output offset=slave bundle=gmem33
    #pragma HLS INTERFACE m_axi port=block1_output offset=slave bundle=gmem34
    #pragma HLS INTERFACE m_axi port=block2_output offset=slave bundle=gmem35
    #pragma HLS INTERFACE m_axi port=block3_output offset=slave bundle=gmem36
    #pragma HLS INTERFACE m_axi port=block4_output offset=slave bundle=gmem37
    
    #pragma HLS INTERFACE m_axi port=output_image offset=slave bundle=gmem38
    
    #pragma HLS INTERFACE s_axilite port=return 
    
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
        block1_output    // Output: 512 channels, 8x8 size
    );
    
    // Block 2: 512 -> 256, 8x8 -> 16x16
    top_block2(
        block1_output,   // Input from block 1
        weights_2,       // Weights
        bias_2,          // Bias
        gamma_2,         // BN gamma
        beta_2,          // BN beta
        running_mean_2,  // BN running mean
        running_var_2,   // BN running var
        block2_output    // Output: 256 channels, 16x16 size
    );
    
    // Block 3: 256 -> 128, 16x16 -> 32x32
    top_block3(
        block2_output,   // Input from block 2
        weights_3,       // Weights
        bias_3,          // Bias
        gamma_3,         // BN gamma
        beta_3,          // BN beta
        running_mean_3,  // BN running mean
        running_var_3,   // BN running var
        block3_output    // Output: 128 channels, 32x32 size
    );
    
    // Block 4: 128 -> 64, 32x32 -> 64x64
    top_block4(
        block3_output,   // Input from block 3
        weights_4,       // Weights
        bias_4,          // Bias
        gamma_4,         // BN gamma
        beta_4,          // BN beta
        running_mean_4,  // BN running mean
        running_var_4,   // BN running var
        block4_output    // Output: 64 channels, 64x64 size
    );
    
    // Block 5: Final ConvTranspose (64 -> 3, 64x64 -> 128x128)
    // This block doesn't have BatchNorm or ReLU
    top_block5(
        block4_output,   // Input from block 4
        weights_5,       // Weights
        bias_5,          // Bias
        output_image     // Final output: 3 channels (RGB), 128x128 size
    );
}
}