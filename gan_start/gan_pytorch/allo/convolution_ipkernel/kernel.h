#ifndef KERNEL_H
#define KERNEL_H

extern "C" {
void top(
  float *input_data,
  float *weight_data,
  float *bias_data,
  float *output_data
);

void kernel_convolution_layer(
  float input[4][3][64][64],
  float weights[16][3][4][4],
  float bias[16],
  float output[4][16][63][63]
);
} // extern "C"

#endif // KERNEL_H
