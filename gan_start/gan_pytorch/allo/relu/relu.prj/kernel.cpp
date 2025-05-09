//===------------------------------------------------------------*- C++ -*-===//
//
// ReLU activation function implementation for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;

extern "C" {

void kernel_relu_layer(
  float input_data[4][16][64][64], 
  float output_data[4][16][64][64]
) {
  // Loop through batch, channels, height, width dimensions
  loop_batch: for (int n = 0; n < 4; n++) {
    loop_channels: for (int c = 0; c < 16; c++) {
      loop_height: for (int h = 0; h < 64; h++) {
        loop_width: for (int w = 0; w < 64; w++) {
          // Get input value at current position
          float input_val = input_data[n][c][h][w];
          
          // ReLU activation: max(0, input_val)
          float output_val = (input_val > 0) ? input_val : 0;
          
          // Store the result
          output_data[n][c][h][w] = output_val;
        }
      }
    }
  }
}

void load_input_data(
  float src_buffer[262144],
  float dest_tensor[4][16][64][64]
) {
  loop_load_batch: for (int n = 0; n < 4; n++) {
    loop_load_channel: for (int c = 0; c < 16; c++) {
      loop_load_height: for (int h = 0; h < 64; h++) {
        loop_load_width: for (int w = 0; w < 64; w++) {
        #pragma HLS pipeline II=1 rewind
          float value = src_buffer[((((n * 65536) + (c * 4096)) + (h * 64)) + w)];
          dest_tensor[n][c][h][w] = value;
        }
      }
    }
  }
}

void store_output_data(
  float src_tensor[4][16][64][64],
  float dest_buffer[262144]
) {
  loop_store_batch: for (int n = 0; n < 4; n++) {
    loop_store_channel: for (int c = 0; c < 16; c++) {
      loop_store_height: for (int h = 0; h < 64; h++) {
        loop_store_width: for (int w = 0; w < 64; w++) {
        #pragma HLS pipeline II=1 rewind
          float value = src_tensor[n][c][h][w];
          dest_buffer[((((n * 65536) + (c * 4096)) + (h * 64)) + w)] = value;
        }
      }
    }
  }
}

void top(
  float *input_data_ptr,
  float *output_data_ptr
) {
  #pragma HLS interface m_axi port=input_data_ptr offset=slave bundle=gmem0
  #pragma HLS interface m_axi port=output_data_ptr offset=slave bundle=gmem1
  
  // Allocate local buffers
  float input_data[4][16][64][64];
  load_input_data(input_data_ptr, input_data);
  
  float output_data[4][16][64][64];
  
  // Execute the ReLU activation
  kernel_relu_layer(input_data, output_data);
  
  // Store results
  store_output_data(output_data, output_data_ptr);
}

} // extern "C"