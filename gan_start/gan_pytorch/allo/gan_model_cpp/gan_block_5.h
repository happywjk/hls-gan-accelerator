#ifndef GAN_BLOCK5_H
#define GAN_BLOCK5_H
#include "gan_block_base.h"

namespace gan_block5 {
    const int IN_CHANNELS = 16;
    const int OUT_CHANNELS = 3;
    const int INPUT_HEIGHT = 64;
    const int INPUT_WIDTH = 64;
    const int PADDING = 1;
    const int STRIDE = 2;
    const int OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) * STRIDE + KERNEL_SIZE - 2*PADDING; // = 128
    const int OUTPUT_WIDTH = (INPUT_WIDTH - 1) * STRIDE + KERNEL_SIZE - 2*PADDING;   // = 128
    const int UNROLL_FACTOR = 1;
}

// Kernel function declaration
extern "C" {
    extern float shared_input_tile[BATCH_SIZE][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    extern float shared_weights[MAX_CHANNELS][MAX_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    extern float shared_bias[MAX_CHANNELS];
    extern float shared_conv_output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE];
    
    void top_block5(
        float *input_data,
        float *weight_data,
        float *bias_data,
        float *output_data
    );
}

#endif // GAN_BLOCK5_H