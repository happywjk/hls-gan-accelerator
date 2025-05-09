#ifndef GAN_BLOCK4_H
#define GAN_BLOCK4_H
#include "gan_block_base.h"

namespace gan_block4 {
    const int IN_CHANNELS = 32;
    const int OUT_CHANNELS = 16;
    const int INPUT_HEIGHT = 32;
    const int INPUT_WIDTH = 32;
    const int PADDING = 1;
    const int STRIDE = 2;
    const int OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) * STRIDE + KERNEL_SIZE - 2*PADDING; // = 64
    const int OUTPUT_WIDTH = (INPUT_WIDTH - 1) * STRIDE + KERNEL_SIZE - 2*PADDING;   // = 64
    const int UNROLL_FACTOR = 1;
    const int OUTPUT_PADDING = (INPUT_WIDTH - 1) * stride + 2*KERNEL_SIZE - 1 - 2*PADDING;
}

// Kernel function declaration
extern "C" {
    extern float shared_input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];
    extern float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    extern float shared_bias[MAX_CHANNELS];
    extern float shared_gamma[MAX_CHANNELS];
    extern float shared_beta[MAX_CHANNELS];
    extern float shared_running_mean[MAX_CHANNELS];
    extern float shared_running_var[MAX_CHANNELS];
    extern float shared_conv_output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];
    extern float shared_bn_output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH];
    
    void top_block4(
        float *input_data,
        float *weight_data,
        float *bias_data,
        float *gamma_data,
        float *beta_data,
        float *running_mean_data,
        float *running_var_data,
        float *output_data
    );
}

#endif // GAN_BLOCK4_H