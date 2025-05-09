#ifndef GAN_MODEL_H
#define GAN_MODEL_H

#include "gan_block_base.h"

// Main kernel entry point
extern "C" {
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
        float *output_image);
}

#endif // GAN_MODEL_H