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
const int UNROLL_FACTOR = 1;
extern "C" {

// Process a single tile of the input
void load_input_tile(
  float *input_data,
  float input_tile[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  int batch_id,
  int row_start,
  int col_start
) {
  #pragma HLS inline off
  const int input_channel_size = INPUT_HEIGHT * INPUT_WIDTH;
  const int input_batch_size = IN_CHANNELS * input_channel_size;

  for (int channel = 0; channel < IN_CHANNELS; channel++) {
    for (int row = 0; row < KERNEL_SIZE; row++) {
      for (int col = 0; col < KERNEL_SIZE; col++) {
        #pragma HLS pipeline II=1 rewind
        
        // Calculate actual input position with padding adjustment
        int in_row_no_stride = row_start + row - KERNEL_SIZE + 1 + PADDING;
        int in_col_no_stride = col_start + col - KERNEL_SIZE + 1 + PADDING;
        int in_row = (in_row_no_stride) / stride;
        int in_col = (in_col_no_stride) / stride;
        
        bool is_orig_input_pos = (in_row_no_stride % stride == 0) && (in_col_no_stride % stride == 0);
        
        // Check if within input bounds
        if (in_row >= 0 && in_row < INPUT_HEIGHT && in_col >= 0 && in_col < INPUT_WIDTH && is_orig_input_pos) {
          // Calculate index with batch offset
          int index = (batch_id * input_batch_size) + 
                      (channel * input_channel_size) + 
                      (in_row * INPUT_WIDTH) + in_col;
          input_tile[channel][row][col] = input_data[index];
        } else {
          // Zero padding for out-of-bounds
          input_tile[channel][row][col] = 0.0f;
        }
      }
    }
  }
}

// Process a single tile of input data
void kernel_convolution_layer_tile(
  float input_tile[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float bias[OUT_CHANNELS],
  float output_tile[OUT_CHANNELS][1][1]
) {
  #pragma HLS inline off

  for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++) {
    float sum = bias[out_channel];
    for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++) {
      #pragma HLS pipeline II=1
      for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
        for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
          sum += weights[in_channel][out_channel][kernel_row][kernel_col] * 
                 input_tile[in_channel][kernel_row][kernel_col];
        }
      }
    }
    
    output_tile[out_channel][0][0] = sum;
  }
}

// Load weights for processing
void load_weights_tile(
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
          int flipped_row = KERNEL_SIZE - 1 - kernel_row;
          int flipped_col = KERNEL_SIZE - 1 - kernel_col;
          weights[in_channel][out_channel][flipped_row][flipped_col] = weight_data[index];
        }
      }
    }
  }
}

// Load bias data
void load_bias_tile(
  float *bias_data,
  float bias[OUT_CHANNELS]
) {
  for (int i = 0; i < OUT_CHANNELS; i++) {
    #pragma HLS pipeline II=1 rewind
    bias[i] = bias_data[i];
  }
}

// Store a single output tile to global memory
void store_output_tile(
  float *output_data,
  float output_tile[OUT_CHANNELS][1][1],
  int batch_id,
  int row,
  int col
) {
  #pragma HLS inline off
  const int output_channel_size = OUTPUT_HEIGHT * OUTPUT_WIDTH;
  const int output_batch_size = OUT_CHANNELS * output_channel_size;

  if (row < OUTPUT_HEIGHT && col < OUTPUT_WIDTH) {
    for (int channel = 0; channel < OUT_CHANNELS; channel++) {
      #pragma HLS pipeline II=1 rewind
      
      // Calculate index in the flattened output array with batch offset
      int index = (batch_id * output_batch_size) + 
                  (channel * output_channel_size) + 
                  (row * OUTPUT_WIDTH) + col;
      output_data[index] = output_tile[channel][0][0];
    }
  }
}

// Load all weights and biases into each of the 8 independent buffers
void load_all_weights_and_biases(
  float *weight_data,
  float *bias_data,
  float weights_0[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_1[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_2[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_3[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_4[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_5[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_6[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_7[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float bias_0[OUT_CHANNELS],
  float bias_1[OUT_CHANNELS],
  float bias_2[OUT_CHANNELS],
  float bias_3[OUT_CHANNELS],
  float bias_4[OUT_CHANNELS],
  float bias_5[OUT_CHANNELS],
  float bias_6[OUT_CHANNELS],
  float bias_7[OUT_CHANNELS]
) {
  // Load weights into each buffer
  load_weights_tile(weight_data, weights_0);
  load_weights_tile(weight_data, weights_1);
  load_weights_tile(weight_data, weights_2);
  load_weights_tile(weight_data, weights_3);
  load_weights_tile(weight_data, weights_4);
  load_weights_tile(weight_data, weights_5);
  load_weights_tile(weight_data, weights_6);
  load_weights_tile(weight_data, weights_7);
  
  // Load biases into each buffer
  load_bias_tile(bias_data, bias_0);
  load_bias_tile(bias_data, bias_1);
  load_bias_tile(bias_data, bias_2);
  load_bias_tile(bias_data, bias_3);
  load_bias_tile(bias_data, bias_4);
  load_bias_tile(bias_data, bias_5);
  load_bias_tile(bias_data, bias_6);
  load_bias_tile(bias_data, bias_7);
}

// Execute convolution kernels in parallel with individual weights and biases
void execute_kernels_in_parallel(
  float input_tile_0[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float input_tile_1[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float input_tile_2[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float input_tile_3[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float input_tile_4[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float input_tile_5[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float input_tile_6[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float input_tile_7[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_0[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_1[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_2[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_3[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_4[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_5[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_6[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float weights_7[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  float bias_0[OUT_CHANNELS],
  float bias_1[OUT_CHANNELS],
  float bias_2[OUT_CHANNELS],
  float bias_3[OUT_CHANNELS],
  float bias_4[OUT_CHANNELS],
  float bias_5[OUT_CHANNELS],
  float bias_6[OUT_CHANNELS],
  float bias_7[OUT_CHANNELS],
  float output_tile_0[OUT_CHANNELS][1][1],
  float output_tile_1[OUT_CHANNELS][1][1],
  float output_tile_2[OUT_CHANNELS][1][1],
  float output_tile_3[OUT_CHANNELS][1][1],
  float output_tile_4[OUT_CHANNELS][1][1],
  float output_tile_5[OUT_CHANNELS][1][1],
  float output_tile_6[OUT_CHANNELS][1][1],
  float output_tile_7[OUT_CHANNELS][1][1]
) {
  #pragma HLS dataflow
  
  kernel_convolution_layer_tile(input_tile_0, weights_0, bias_0, output_tile_0);
  kernel_convolution_layer_tile(input_tile_1, weights_1, bias_1, output_tile_1);
  kernel_convolution_layer_tile(input_tile_2, weights_2, bias_2, output_tile_2);
  kernel_convolution_layer_tile(input_tile_3, weights_3, bias_3, output_tile_3);
  kernel_convolution_layer_tile(input_tile_4, weights_4, bias_4, output_tile_4);
  kernel_convolution_layer_tile(input_tile_5, weights_5, bias_5, output_tile_5);
  kernel_convolution_layer_tile(input_tile_6, weights_6, bias_6, output_tile_6);
  kernel_convolution_layer_tile(input_tile_7, weights_7, bias_7, output_tile_7);
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
  
  // Allocate independent weights and biases for each batch
  float weights_0[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights_1[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights_2[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights_3[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights_4[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights_5[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights_6[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float weights_7[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  
  float bias_0[OUT_CHANNELS];
  float bias_1[OUT_CHANNELS];
  float bias_2[OUT_CHANNELS];
  float bias_3[OUT_CHANNELS];
  float bias_4[OUT_CHANNELS];
  float bias_5[OUT_CHANNELS];
  float bias_6[OUT_CHANNELS];
  float bias_7[OUT_CHANNELS];
  
  // Allocate batch-specific tile buffers
  float input_tile_0[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float input_tile_1[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float input_tile_2[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float input_tile_3[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float input_tile_4[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float input_tile_5[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float input_tile_6[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  float input_tile_7[IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  
  float output_tile_0[OUT_CHANNELS][1][1];
  float output_tile_1[OUT_CHANNELS][1][1];
  float output_tile_2[OUT_CHANNELS][1][1];
  float output_tile_3[OUT_CHANNELS][1][1];
  float output_tile_4[OUT_CHANNELS][1][1];
  float output_tile_5[OUT_CHANNELS][1][1];
  float output_tile_6[OUT_CHANNELS][1][1];
  float output_tile_7[OUT_CHANNELS][1][1];
  
  // Partition weights and bias arrays for parallel access
  #pragma HLS array_partition variable=weights_0 complete dim=2
  #pragma HLS array_partition variable=weights_1 complete dim=2
  #pragma HLS array_partition variable=weights_2 complete dim=2
  #pragma HLS array_partition variable=weights_3 complete dim=2
  #pragma HLS array_partition variable=weights_4 complete dim=2
  #pragma HLS array_partition variable=weights_5 complete dim=2
  #pragma HLS array_partition variable=weights_6 complete dim=2
  #pragma HLS array_partition variable=weights_7 complete dim=2
  
  #pragma HLS array_partition variable=bias_0 complete dim=1
  #pragma HLS array_partition variable=bias_1 complete dim=1
  #pragma HLS array_partition variable=bias_2 complete dim=1
  #pragma HLS array_partition variable=bias_3 complete dim=1
  #pragma HLS array_partition variable=bias_4 complete dim=1
  #pragma HLS array_partition variable=bias_5 complete dim=1
  #pragma HLS array_partition variable=bias_6 complete dim=1
  #pragma HLS array_partition variable=bias_7 complete dim=1
  
  // Assign each input/output tile to different memory banks
  #pragma HLS bind_storage variable=input_tile_0 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=input_tile_1 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=input_tile_2 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=input_tile_3 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=input_tile_4 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=input_tile_5 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=input_tile_6 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=input_tile_7 type=RAM_T2P impl=BRAM
  
  // Assign each weight/bias buffer to different memory banks
  #pragma HLS bind_storage variable=weights_0 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=weights_1 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=weights_2 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=weights_3 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=weights_4 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=weights_5 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=weights_6 type=RAM_T2P impl=BRAM
  #pragma HLS bind_storage variable=weights_7 type=RAM_T2P impl=BRAM
  
  // Load weights and biases into each buffer
  load_all_weights_and_biases(
    weight_data, bias_data,
    weights_0, weights_1, weights_2, weights_3,
    weights_4, weights_5, weights_6, weights_7,
    bias_0, bias_1, bias_2, bias_3,
    bias_4, bias_5, bias_6, bias_7
  );
  
  // Process each output position
  for (int out_row = 0; out_row < OUTPUT_HEIGHT; out_row++) { 
    for (int out_col = 0; out_col < OUTPUT_WIDTH; out_col++) {
      // 1. First load all batch tiles
      load_input_tile(input_data, input_tile_0, 0, out_row, out_col);
      load_input_tile(input_data, input_tile_1, 1, out_row, out_col);
      load_input_tile(input_data, input_tile_2, 2, out_row, out_col);
      load_input_tile(input_data, input_tile_3, 3, out_row, out_col);
      load_input_tile(input_data, input_tile_4, 4, out_row, out_col);
      load_input_tile(input_data, input_tile_5, 5, out_row, out_col);
      load_input_tile(input_data, input_tile_6, 6, out_row, out_col);
      load_input_tile(input_data, input_tile_7, 7, out_row, out_col);
      
      // 2. Process all batch tiles in parallel with dataflow
      execute_kernels_in_parallel(
        input_tile_0, input_tile_1, input_tile_2, input_tile_3, 
        input_tile_4, input_tile_5, input_tile_6, input_tile_7,
        weights_0, weights_1, weights_2, weights_3,
        weights_4, weights_5, weights_6, weights_7,
        bias_0, bias_1, bias_2, bias_3,
        bias_4, bias_5, bias_6, bias_7,
        output_tile_0, output_tile_1, output_tile_2, output_tile_3,
        output_tile_4, output_tile_5, output_tile_6, output_tile_7
      );
      
      // 3. Store all output tiles
      store_output_tile(output_data, output_tile_0, 0, out_row, out_col);
      store_output_tile(output_data, output_tile_1, 1, out_row, out_col);
      store_output_tile(output_data, output_tile_2, 2, out_row, out_col);
      store_output_tile(output_data, output_tile_3, 3, out_row, out_col);
      store_output_tile(output_data, output_tile_4, 4, out_row, out_col);
      store_output_tile(output_data, output_tile_5, 5, out_row, out_col);
      store_output_tile(output_data, output_tile_6, 6, out_row, out_col);
      store_output_tile(output_data, output_tile_7, 7, out_row, out_col);
    }
  }
}

}