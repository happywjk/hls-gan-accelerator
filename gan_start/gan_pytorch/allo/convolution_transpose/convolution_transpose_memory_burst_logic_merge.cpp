#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
#include <cstring> 
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
const int burst_size = 512;
extern "C" {


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



void top(
  float *input_data,
  float *weight_data,
  float *bias_data,
  float *output_data
) {
    #pragma HLS interface m_axi port=input_data offset=slave bundle=gmem0 max_read_burst_length=256 max_write_burst_length=256
  #pragma HLS interface m_axi port=weight_data offset=slave bundle=gmem1
  #pragma HLS interface m_axi port=bias_data offset=slave bundle=gmem2
  #pragma HLS interface m_axi port=output_data offset=slave bundle=gmem3
  
  // Allocate on-chip buffers (BRAM)
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  #pragma HLS array_partition variable=weights dim=1 complete
    #pragma HLS array_partition variable=weights dim=3 complete
    #pragma HLS array_partition variable=weights dim=4 complete
  float bias[OUT_CHANNELS];
  const int input_channel_size = INPUT_HEIGHT * INPUT_WIDTH;
  const int input_batch_size = IN_CHANNELS * input_channel_size;
  const int output_channel_size = OUTPUT_HEIGHT * OUTPUT_WIDTH;
  const int output_batch_size = OUT_CHANNELS * output_channel_size;
  float input_burst[128];
  #pragma HLS array_partition variable=input_burst cyclic factor=16
  
  // Load weights and biases once (assuming they fit in BRAM)
  load_weights_tile(weight_data, weights);
  load_bias_tile(bias_data, bias);
  
  // Process each output position
  for (int row_start = 0; row_start < OUTPUT_HEIGHT; row_start++) { 
    for (int col_start = 0; col_start < OUTPUT_WIDTH; col_start++) {

        for (int batch = 0; batch < BATCH_SIZE; batch++) {
            int index = (row_start*26214+ col_start*409 + batch * 51);
            memcpy(input_burst, &input_data[index], 256 * sizeof(float));
            memcpy(input_burst+256, &input_data[index+256], 256 * sizeof(float));
            for (int out_channel = 0; out_channel < OUT_CHANNELS; out_channel++) {
            #pragma HLS pipeline II=1
              float sum = bias[out_channel];
              for (int in_channel = 0; in_channel < IN_CHANNELS; in_channel++) {
                for (int kernel_row = 0; kernel_row < KERNEL_SIZE; kernel_row++) {
                  for (int kernel_col = 0; kernel_col < KERNEL_SIZE; kernel_col++) {
                    int in_row_no_stride = row_start + kernel_row - KERNEL_SIZE +1 + PADDING;
                    int in_col_no_stride = col_start + kernel_col - KERNEL_SIZE +1 + PADDING;
                    int in_row = (in_row_no_stride) / stride;
                    int in_col = (in_col_no_stride) / stride;
                    bool is_orig_input_pos = (in_row_no_stride % stride == 0) && (in_col_no_stride % stride == 0);
                    
                    if (in_row >= 0 && in_row < INPUT_HEIGHT && in_col >= 0 && in_col < INPUT_WIDTH && is_orig_input_pos){
                        index_1 =  index_1 +1;
                        sum += weights[in_channel][out_channel][kernel_row][kernel_col] * input_burst[index];
                    } 
        
                  }
                }
              }
              int index_output = (batch * output_batch_size) + 
                            (out_channel * output_channel_size) + 
                            (row_start * OUTPUT_WIDTH) + col_start;
              
              output_data[index_output] = sum;
            }
          }
    }
  }
}

}