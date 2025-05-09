#ifndef BATCHNORM_H
#define BATCHNORM_H

#include "gan_block_base.h"

// Process a single tile through BatchNorm2d
void kernel_batchnorm_layer_tile(
    float input_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
    float gamma[MAX_CHANNELS],
    float beta[MAX_CHANNELS],
    float running_mean[MAX_CHANNELS],
    float running_var[MAX_CHANNELS],
    float output_tile[BATCH_SIZE][MAX_CHANNELS][MAX_TILE_SIZE][MAX_TILE_SIZE],
    int batch_size,
    int channels,
    int tile_height,
    int tile_width
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
);

#endif // BATCHNORM_H