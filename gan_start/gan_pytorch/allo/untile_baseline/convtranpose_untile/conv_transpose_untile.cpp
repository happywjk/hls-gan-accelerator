#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

// Global parameters
const int BATCH_SIZE = 8;
const int IN_CHANNELS = 32;
const int OUT_CHANNELS = 16;
const int KERNEL_SIZE = 4;
const int INPUT_HEIGHT = 32;
const int INPUT_WIDTH = 32;
const int PADDING = 1;
const int stride = 2;
const int OUTPUT_HEIGHT = (INPUT_HEIGHT - 1) * stride + KERNEL_SIZE - 2*PADDING;
const int OUTPUT_WIDTH = (INPUT_WIDTH - 1) * stride + KERNEL_SIZE - 2*PADDING;
const int OUTPUT_PADDING = (INPUT_WIDTH - 1) * stride + 2*KERNEL_SIZE - 1 - 2*PADDING;
const int UNROLL_FACTOR = 1;

extern "C" {

  void kernel_convolution_transpose_layer(
    float input[BATCH_SIZE][IN_CHANNELS][OUTPUT_PADDING][OUTPUT_PADDING],
    float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
    float bias[OUT_CHANNELS],
    float output[BATCH_SIZE][OUT_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH]
  ) {
    // Process each output position
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
      for (int out_c = 0; out_c < OUT_CHANNELS; out_c++) {
        for (int out_h = 0; out_h < OUTPUT_HEIGHT; out_h++) {
          for (int out_w = 0; out_w < OUTPUT_WIDTH; out_w++) {
            // Initialize with bias
            float sum = bias[out_c];
            
            // Perform standard convolution operation
            for (int in_c = 0; in_c < IN_CHANNELS; in_c++) {
              #pragma HLS unroll factor=UNROLL_FACTOR
              for (int k_h = 0; k_h < KERNEL_SIZE; k_h++) {
                #pragma HLS unroll factor=UNROLL_FACTOR
                for (int k_w = 0; k_w < KERNEL_SIZE; k_w++) {
                  #pragma HLS unroll factor=UNROLL_FACTOR               
                  sum += weights[in_c][out_c][k_h][k_w] * 
                  input[batch][in_c][out_h+k_h][out_w+k_w];
                
                }
              }
            }           
            // Store result
            output[batch][out_c][out_h][out_w] = sum;
          }
        }
      }
    }
  }


// Load the entire input

void load_input(
  float *input_data,
  float input[BATCH_SIZE][IN_CHANNELS][OUTPUT_PADDING][OUTPUT_PADDING]
) {
  const int input_channel_size = INPUT_HEIGHT * INPUT_WIDTH;
  const int input_batch_size = IN_CHANNELS * input_channel_size;

  // Initialize the entire input array with zeros (for padding)
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < IN_CHANNELS; channel++) {
      for (int h = 0; h < OUTPUT_PADDING; h++) {
        for (int w = 0; w < OUTPUT_PADDING; w++) {
          input[batch][channel][h][w] = 0.0f;
        }
      }
    }
  }

  // Load actual input data with appropriate striding and padding
  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < IN_CHANNELS; channel++) {
      for (int in_h = 0; in_h < INPUT_HEIGHT; in_h++) {
        for (int in_w = 0; in_w < INPUT_WIDTH; in_w++) {
          #pragma HLS pipeline II=1 rewind
          
          // Calculate index in the flattened input array
          int index = (batch * input_batch_size) + 
                      (channel * input_channel_size) + 
                      (in_h * INPUT_WIDTH) + in_w;
          
          // Calculate the corresponding position in the output array for transposed convolution
          int out_h = in_h * stride  + KERNEL_SIZE -1-PADDING;
          int out_w = in_w * stride  + KERNEL_SIZE -1-PADDING;
          
          input[batch][channel][out_h][out_w] = input_data[index];
          // Only copy if within output bounds (considering padding)
        }
      }
    }
  }
}

// Load weights for processing
void load_weights(
  float *weight_data,
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE]
) {
  const int weight_channel_size = KERNEL_SIZE * KERNEL_SIZE;
  const int weight_out_channel_size = OUT_CHANNELS * weight_channel_size;

  for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++) {
    for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++) {
      for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
        for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
          #pragma HLS pipeline II=1 rewind
          int index = (in_channel * weight_out_channel_size) + 
                      (out_channel * weight_channel_size) + 
                      (kernel_row * KERNEL_SIZE) + kernel_col;
          
          // Flip the kernel for transposed convolution
          int flipped_row = KERNEL_SIZE - 1 - kernel_row;
          int flipped_col = KERNEL_SIZE - 1 - kernel_col;
          weights[in_channel][out_channel][flipped_row][flipped_col] = weight_data[index];
        }
      }
    }
  }
}

// Load bias data
void load_bias(
  float *bias_data,
  float bias[OUT_CHANNELS]
) {
  for (int i = 0; i < OUT_CHANNELS; i++) {
    #pragma HLS pipeline II=1 rewind
    bias[i] = bias_data[i];
  }
}

// Store the entire output
void store_output(
  float *output_data,
  float output[BATCH_SIZE][OUT_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH]
) {
  const int output_channel_size = OUTPUT_HEIGHT * OUTPUT_WIDTH;
  const int output_batch_size = OUT_CHANNELS * output_channel_size;

  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < OUT_CHANNELS; channel++) {
      for (int row = 0; row < OUTPUT_HEIGHT; row++) {
        for (int col = 0; col < OUTPUT_WIDTH; col++) {
          #pragma HLS pipeline II=1 rewind
          int index = (batch * output_batch_size) + 
                      (channel * output_channel_size) + 
                      (row * OUTPUT_WIDTH) + col;
          output_data[index] = output[batch][channel][row][col];
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
  
  // Allocate on-chip buffers for full data
  float input[BATCH_SIZE][IN_CHANNELS][OUTPUT_PADDING][OUTPUT_PADDING];
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float bias[OUT_CHANNELS];
  float output[BATCH_SIZE][OUT_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH];
  
  // Load full input, weights, and biases
  load_input(input_data, input);
  load_weights(weight_data, weights);
  load_bias(bias_data, bias);
  
  // Process the entire input at once
  kernel_convolution_transpose_layer(input, weights, bias, output);
  
  // Store the entire output
  store_output(output_data, output);
}

}