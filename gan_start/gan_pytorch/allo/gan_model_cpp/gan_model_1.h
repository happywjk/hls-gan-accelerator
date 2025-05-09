#ifndef GAN_MODEL_H
#define GAN_MODEL_H

#include "gan_block_base.h"

// Main kernel entry point
extern "C" {
void top(
    // Input noise vector
    float *noise_input,
    
    // Block 0 parameters
    float *weights_0,
    float *bias_0,
    float *gamma_0,
    float *beta_0,
    float *running_mean_0,
    float *running_var_0,
    
    // Block 1 parameters
    float *weights_1,
    float *bias_1,
    float *gamma_1,
    float *beta_1,
    float *running_mean_1,
    float *running_var_1,
    
    // Final output
    float *output_image
);
}

#endif // GAN_MODEL_H