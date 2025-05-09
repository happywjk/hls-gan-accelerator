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



// Load a single input tile from global memory with proper padding
void load_input_tile(
  float *input_data,
  float input_tile[BATCH_SIZE][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],
  int row_start,
  int col_start
) {
  #pragma HLS inline off 
  const int input_channel_size = INPUT_HEIGHT * INPUT_WIDTH;
  const int input_batch_size = IN_CHANNELS * input_channel_size;

  for (int batch = 0; batch < BATCH_SIZE; batch++) {
    for (int channel = 0; channel < IN_CHANNELS; channel++) {
      for (int row = 0; row < KERNEL_SIZE; row++) {
        for (int col = 0; col < KERNEL_SIZE; col++) {
          #pragma HLS pipeline II=1 rewind
          
          // Calculate actual input position with padding adjustment
          int in_row_no_stride = row_start + row - KERNEL_SIZE +1 + PADDING;
          int in_col_no_stride = col_start + col - KERNEL_SIZE +1 + PADDING;
          int in_row = (in_row_no_stride) / stride;
          int in_col = (in_col_no_stride) / stride;
          
          bool is_orig_input_pos = (in_row_no_stride % stride == 0) && (in_col_no_stride % stride == 0);
          // Check if within input bounds
          if (in_row >= 0 && in_row < INPUT_HEIGHT && in_col >= 0 && in_col < INPUT_WIDTH && is_orig_input_pos) {
            // Calculate index in the flattened input array
            int index = (batch * input_batch_size) + 
                        (channel * input_channel_size) + 
                        (in_row * INPUT_WIDTH) + in_col;
            input_tile[batch][channel][row][col] = input_data[index];
          } else {
            // Zero padding for out-of-bounds
            input_tile[batch][channel][row][col] = 0.0f;
          }
        }
      }
    }
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
  float output_tile[BATCH_SIZE][OUT_CHANNELS][1][1],
  int row,
  int col
) {
  #pragma HLS inline off
  const int output_channel_size = OUTPUT_HEIGHT * OUTPUT_WIDTH;
  const int output_batch_size = OUT_CHANNELS * output_channel_size;

  // Check if the output tile position is within bounds
  if (row < OUTPUT_HEIGHT && col < OUTPUT_WIDTH) {
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
      for (int channel = 0; channel < OUT_CHANNELS; channel++) {
        #pragma HLS pipeline II=1 rewind
        
        // Calculate index in the flattened output array
        int index = (batch * output_batch_size) + 
                    (channel * output_channel_size) + 
                    (row * OUTPUT_WIDTH) + col;
        output_data[index] = output_tile[batch][channel][0][0];
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
  
  // On-chip buffers for weights and bias (weight-stationary design)
  float weights[IN_CHANNELS][OUT_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
  #pragma HLS array_partition variable=weights complete dim=3
  #pragma HLS array_partition variable=weights complete dim=4
  
  float bias[OUT_CHANNELS];
  #pragma HLS array_partition variable=bias complete
  
  // Load weights and biases once (weights stay stationary)
  load_weights_tile(weight_data, weights);
  load_bias_tile(bias_data, bias);
  
  // On-chip buffer for partial sums
  float output_buffer[BATCH_SIZE][OUT_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH];
  #pragma HLS array_partition variable=output_buffer complete dim=2
  
  // Initialize output buffer with bias values
  for (int b = 0; b < BATCH_SIZE; b++) {
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
      for (int oh = 0; oh < OUTPUT_HEIGHT; oh++) {
        for (int ow = 0; ow < OUTPUT_WIDTH; ow++) {
          #pragma HLS pipeline II=1
          output_buffer[b][oc][oh][ow] = bias[oc];
        }
      }
    }
  }
  
  // Systolic array processing across entire feature map
  // Process each input channel sequentially
  systolic1:
  for (int ic = 0; ic < IN_CHANNELS; ic++) {
    #pragma HLS LOOP_TRIPCOUNT min=IN_CHANNELS max=IN_CHANNELS
    
    // For each input-output channel pair, we process the entire spatial dimensions
    systolic2:
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
      // Process in a wave-front pattern across the entire output feature map
      // This creates a systolic flow of data through the processing elements
      
      // Outer loops for wave-front pattern
      for (int wave = 0; wave < OUTPUT_HEIGHT + OUTPUT_WIDTH - 1; wave++) {
        #pragma HLS PIPELINE II=1
        
        systolic3:
        for (int oh = 0; oh < OUTPUT_HEIGHT; oh++) {
          for (int ow = 0; ow < OUTPUT_WIDTH; ow++) {
            #pragma HLS UNROLL factor=UNROLL_FACTOR
            
            // Check if this PE is active in the current wave
            if (oh + ow == wave) {
              // For each output position, compute convolution with the kernel
              float partial_sum = 0.0f;
              
              for (int kr = 0; kr < KERNEL_SIZE; kr++) {
                for (int kc = 0; kc < KERNEL_SIZE; kc++) {
                  #pragma HLS UNROLL
                  
                  // Calculate input position with padding and stride
                  int in_row = oh * stride - PADDING + kr;
                  int in_col = ow * stride - PADDING + kc;
                  
                  // Process all batches for this position
                  for (int b = 0; b < BATCH_SIZE; b++) {
                    // Check boundary conditions
                    float input_val = 0.0f;
                    if (in_row >= 0 && in_row < INPUT_HEIGHT && 
                        in_col >= 0 && in_col < INPUT_WIDTH) {
                      // Calculate index in flattened input array
                      int input_idx = (b * IN_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH) + 
                                     (ic * INPUT_HEIGHT * INPUT_WIDTH) + 
                                     (in_row * INPUT_WIDTH) + in_col;
                      input_val = input_data[input_idx];
                    }
                    
                    // In a weight-stationary design, weights stay fixed at each PE
                    float weight_val = weights[ic][oc][kr][kc];
                    
                    // Accumulate partial product into the output buffer
                    output_buffer[b][oc][oh][ow] += input_val * weight_val;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  
  // Store final results to global memory
  for (int b = 0; b < BATCH_SIZE; b++) {
    for (int oc = 0; oc < OUT_CHANNELS; oc++) {
      for (int oh = 0; oh < OUTPUT_HEIGHT; oh++) {
        for (int ow = 0; ow < OUTPUT_WIDTH; ow++) {
          #pragma HLS PIPELINE II=1
          int output_idx = (b * OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) + 
                          (oc * OUTPUT_HEIGHT * OUTPUT_WIDTH) + 
                          (oh * OUTPUT_WIDTH) + ow;
          output_data[output_idx] = output_buffer[b][oc][oh][ow];
        }
      }
    }
  }
}

}