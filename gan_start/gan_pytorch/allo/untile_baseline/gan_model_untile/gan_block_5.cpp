#include "gan_block_5.h"
#include "gan_block_base.h"
#include "conv_transpose.h"
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
using namespace std;

// External buffer declarations with correct dimensions for untiled approach
extern float shared_input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];
extern float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
extern float shared_bias[MAX_CHANNELS];
extern float shared_conv_output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];

extern "C" {
  void top_block5(
    float *input_data,
    float *weight_data,
    float *bias_data,
    float *output_data
  ) {
    #pragma HLS interface m_axi port=input_data offset=slave bundle=gmem0
    #pragma HLS interface m_axi port=weight_data offset=slave bundle=gmem1
    #pragma HLS interface m_axi port=bias_data offset=slave bundle=gmem2
    #pragma HLS interface m_axi port=output_data offset=slave bundle=gmem3
    
    // Load all parameters once
    load_weights(weight_data, shared_weights, gan_block5::IN_CHANNELS, gan_block5::OUT_CHANNELS);
    load_bias(bias_data, shared_bias, gan_block5::OUT_CHANNELS);
    
    // 1. Load the entire input
    load_input(input_data, shared_input, 
               gan_block5::IN_CHANNELS, 
               gan_block5::INPUT_HEIGHT, 
               gan_block5::INPUT_WIDTH,
               gan_block5::OUTPUT_HEIGHT,
               gan_block5::OUTPUT_WIDTH,
               gan_block5::OUTPUT_PADDING,
               gan_block5::PADDING, 
               gan_block5::STRIDE);
    
    // 2. Process the entire input through ConvTranspose2d
    kernel_convolution_transpose_layer(
        shared_input, shared_weights, shared_bias, shared_conv_output,
        gan_block5::IN_CHANNELS, 
        gan_block5::OUT_CHANNELS,
        gan_block5::INPUT_HEIGHT,
        gan_block5::INPUT_WIDTH,
        gan_block5::OUTPUT_HEIGHT,
        gan_block5::OUTPUT_WIDTH,
        gan_block5::STRIDE,
        gan_block5::PADDING);
    
    // 3. Apply tanh activation (instead of BatchNorm+ReLU used in earlier blocks)
    kernel_tanh_layer(
        shared_conv_output, shared_conv_output,
        BATCH_SIZE, 
        gan_block5::OUT_CHANNELS,
        gan_block5::OUTPUT_HEIGHT, 
        gan_block5::OUTPUT_WIDTH);
    
    // 4. Store the entire output
    store_output(
        output_data, shared_conv_output,
        BATCH_SIZE, 
        gan_block5::OUT_CHANNELS,
        gan_block5::OUTPUT_HEIGHT, 
        gan_block5::OUTPUT_WIDTH);
  }
}