#ifndef GAN_BLOCK1_H
#define GAN_BLOCK1_H
#include "gan_block_base.h"
namespace gan_block1 {
    const int IN_CHANNELS = 256;
    const int OUT_CHANNELS = 128;
    const int INPUT_HEIGHT = 4;
    const int INPUT_WIDTH = 4;
    const int PADDING = 1;
    const int STRIDE = 2;
    const int OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) * STRIDE + KERNEL_SIZE - 2*PADDING; // = 8
    const int OUTPUT_WIDTH = (INPUT_WIDTH - 1) * STRIDE + KERNEL_SIZE - 2*PADDING;   // = 8
    const int UNROLL_FACTOR = 1;
    const int OUTPUT_PADDING = (INPUT_WIDTH - 1) * stride + 2*KERNEL_SIZE - 1 - 2*PADDING;
}
// Add to each gan_block header file:

// Parameters for Block 1


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
    void top_block1(
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

#endif // GAN_BLOCK1_H