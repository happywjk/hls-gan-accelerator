//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

extern "C" {

void kernel_convolution_layer(
  float input[4][3][64][64],
  float weights[16][3][4][4],
  float bias[16],
  float output[4][16][63][63]
) {
  float padded_input[4][3][66][66];

  // Initialize padded input
  for (int batch = 0; batch < 4; batch++) {
    for (int channel = 0; channel < 3; channel++) {
      #pragma HLS unroll factor=1 // Prevent unrolling small outer loops
      for (int row = 0; row < 66; row++) {
        for (int col = 0; col < 66; col++) {
          padded_input[batch][channel][row][col] = 0.0f;
        }
      }
    }
  }

  // Copy with padding
  for (int batch = 0; batch < 4; batch++) {
    for (int channel = 0; channel < 3; channel++) {
      #pragma HLS unroll factor=1
      for (int row = 0; row < 64; row++) {
        for (int col = 0; col < 64; col++) {
          padded_input[batch][channel][row + 1][col + 1] = input[batch][channel][row][col];
        }
      }
    }
  }

  // Convolution
  for (int batch = 0; batch < 4; batch++) {
    for (int out_channel = 0; out_channel < 16; out_channel++) {
      for (int out_row = 0; out_row < 63; out_row++) {
        for (int out_col = 0; out_col < 63; out_col++) {
          float sum = bias[out_channel];
          for (int in_channel = 0; in_channel < 3; in_channel++) {
            #pragma HLS unroll factor=1
            for (int kernel_row = 0; kernel_row < 4; kernel_row++) {
              #pragma HLS unroll factor=1
              for (int kernel_col = 0; kernel_col < 4; kernel_col++) {
                #pragma HLS unroll factor=1
                sum += weights[out_channel][in_channel][kernel_row][kernel_col] * 
                        padded_input[batch][in_channel][out_row + kernel_row][out_col + kernel_col];
              }
            }
          }
          output[batch][out_channel][out_row][out_col] = sum;
        }
      }
    }
  }
}

void load_input(float input_data[49152], float input[4][3][64][64]) {
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 64; k++) {
        for (int l = 0; l < 64; l++) {
          #pragma HLS pipeline II=1 rewind
          input[i][j][k][l] = input_data[(i * 12288) + (j * 4096) + (k * 64) + l];
        }
      }
    }
  }
}

void load_weights(float weight_data[768], float weights[16][3][4][4]) {
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 4; l++) {
          #pragma HLS pipeline II=1 rewind
          weights[i][j][k][l] = weight_data[(i * 48) + (j * 16) + (k * 4) + l];
        }
      }
    }
  }
}

void load_bias(float bias_data[16], float bias[16]) {
  for (int i = 0; i < 16; i++) {
    #pragma HLS pipeline II=1 rewind
    bias[i] = bias_data[i];
  }
}

// void load_output(float output_data[254016], float output[4][16][63][63]) {
//   for (int i = 0; i < 4; i++) {
//     for (int j = 0; j < 16; j++) {
//       for (int k = 0; k < 63; k++) {
//         for (int l = 0; l < 63; l++) {
//           #pragma HLS pipeline II=1 rewind
//           output[i][j][k][l] = output_data[(i * 63504) + (j * 3969) + (k * 63) + l];
//         }
//       }
//     }
//   }
// }

void store_output(float output[4][16][63][63], float output_data[254016]) {

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 63; k++) {
        for (int l = 0; l < 63; l++) {
          #pragma HLS pipeline II=1 rewind
          output_data[(i * 63504) + (j * 3969) + (k * 63) + l] = output[i][j][k][l];
        }
      }
    }
  }
}

void top(
  float *input_data,
  float *weight_data,
  float *bias_data,
  float *output_data
) {
  #pragma HLS interface m_axi port=input_data offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=weight_data offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=bias_data offset=slave bundle=gmem2
  #pragma HLS interface m_axi port=output_data offset=slave bundle=gmem3

  float input[4][3][64][64];
  load_input(input_data, input);

  float weights[16][3][4][4];
  load_weights(weight_data, weights);

  float bias[16];
  load_bias(bias_data, bias);

  float output[4][16][63][63];
  // load_output(output_data, output);

  kernel_convolution_layer(input, weights, bias, output);

  store_output(output, output_data);
}

} // extern "C"
