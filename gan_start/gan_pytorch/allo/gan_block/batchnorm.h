#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include "gan_block.h"

// Apply batch normalization to a tile
void kernel_batchnorm_layer_tile(
    float input_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
    float gamma[OUT_CHANNELS],
    float beta[OUT_CHANNELS],
    float running_mean[OUT_CHANNELS],
    float running_var[OUT_CHANNELS],
    float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1]
);

// Load batch normalization parameters
void load_batchnorm_params(
    float *gamma_data,
    float *beta_data,
    float *running_mean_data,
    float *running_var_data,
    float gamma[OUT_CHANNELS],
    float beta[OUT_CHANNELS],
    float running_mean[OUT_CHANNELS],
    float running_var[OUT_CHANNELS]
);

#endif // BATCH_NORM_H