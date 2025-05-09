#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "gan_block_base.h"

// Process a single tile through BatchNorm2d
void kernel_batchnorm_layer(
    float input[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
    float gamma[MAX_CHANNELS],
    float beta[MAX_CHANNELS],
    float running_mean[MAX_CHANNELS],
    float running_var[MAX_CHANNELS],
    float output[BATCH_SIZE][MAX_CHANNELS][MAX_OUTPUT_HEIGHT][MAX_OUTPUT_WIDTH],
    int batch_size,
    int channels,
    int height,
    int width
  );

// Load batchnorm parameters from global memory
void load_batchnorm_params(
    float *gamma_data,
    float *beta_data,
    float *running_mean_data,
    float *running_var_data,
    float gamma[MAX_CHANNELS],
    float beta[MAX_CHANNELS],
    float running_mean[MAX_CHANNELS],
    float running_var[MAX_CHANNELS],
    int channels
  ) ;

#endif // BATCHNORM_H