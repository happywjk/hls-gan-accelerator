#ifndef GEN_BLOCK_H
#define GEN_BLOCK_H

// Global parameters
const int BATCH_SIZE = 4;
const int IN_CHANNELS = 3;
const int OUT_CHANNELS = 16;
const int KERNEL_SIZE = 4;
const int INPUT_HEIGHT = 64;
const int INPUT_WIDTH = 64;
const int PADDING = 0;
const int stride = 2;
const int OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) * stride + KERNEL_SIZE - 2*PADDING; 
const int OUTPUT_WIDTH = (INPUT_WIDTH - 1) * stride + KERNEL_SIZE - 2*PADDING;
const int UNROLL_FACTOR = 1;

// Main function declaration
extern "C" {
    void top(
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

#endif // GEN_BLOCK_H