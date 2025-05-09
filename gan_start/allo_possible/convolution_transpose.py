import numpy as np
import allo
from allo.ir.types import int32,float32
from allo import dsl
from torch import nn
import torch
import time
from allo.ir.utils import MockBuffer

        # Forward pass for linear layer
        # input = add_padding(input,padding)
        # for i, j,k,z in dsl.grid(batch, cout,hout,wout):
        #     out[i, j,k,z] = Bias[j]  # Initialize with bias
        #     for m in range(cin):
        #         n = k * stride  # Compute corresponding input row
        #         l = z * stride  # Compute corresponding input column
        #         for p in range(kernel_height):
        #             for q in range(kernel_width):
        #                 out[i, j,k,z] += Weight[j,m,p,q]* input[i,m,n+p,l+q]


# %%
# unoptimized version
def ConvolutionTranspose_layer_unoptimized():
    # Parameters for the transposed convolution
    concrete_type, batch_size, channel_in, channel_out, height, width, kernel_height, kernel_width, stride, padding = float32, 4, 3, 16, 64, 64, 4, 4, 1, 1
    
    # Calculate output dimensions for transposed convolution
    # For transposed convolution: output_size = (input_size - 1) * stride + kernel_size - 2 * padding
    height_out = (height - 1) * stride + kernel_height - 2 * padding
    width_out = (width - 1) * stride + kernel_width - 2 * padding
    
    def kernel_convolution_transpose_layer[Ty, batch, cin, cout, height, width, kernel_height, kernel_width, hout, wout, stride, padding](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cin, cout, kernel_height, kernel_width]", Bias: "Ty[cout]", out: "Ty[batch, cout, hout, wout]"
            ):
        # Initialize output with bias
        for i, j, k, z in dsl.grid(batch, cout, hout, wout):
            out[i, j, k, z] = Bias[j]
        
        # Perform transposed convolution
        for i, j, h, w in dsl.grid(batch, cin, height, width):
            for m, p, q in allo.reduction(cout, kernel_height, kernel_width):
                h_out = h * stride + p - padding
                w_out = w * stride + q - padding
                
                # Only update output if coordinates are valid
                if 0 <= h_out < hout and 0 <= w_out < wout:
                    out[i, m, h_out, w_out] += Weight[j, m, p, q] * input[i, j, h, w]
    
    s_linear = allo.customize(kernel_convolution_transpose_layer, 
                              instantiate=[concrete_type, batch_size, channel_in, channel_out, height, width, 
                                          kernel_height, kernel_width, height_out, width_out, stride, padding])

    def top[Ty, batch, cin, cout, height, width, kernel_height, kernel_width, hout, wout, stride, padding](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cin, cout, kernel_height, kernel_width]", Bias: "Ty[cout]", out: "Ty[batch, cout, hout, wout]"
    ):
        kernel_convolution_transpose_layer[float32, batch_size, channel_in, channel_out, height, width, 
                                           kernel_height, kernel_width, height_out, width_out, stride, padding](input, Weight, Bias, out)

    s = allo.customize(top, instantiate=[concrete_type, batch_size, channel_in, channel_out, height, width, 
                                         kernel_height, kernel_width, height_out, width_out, stride, padding])
    s.compose(s_linear)

    return s.build(target="vitis_hls", mode="sw_emu", project="convolution_transpose_ipkernel.prj")

# %%
# pipeline version
def Convolution_layer_pipeline():
    # concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,height_out,widthout,stride = int32,16,3,128,258,258,4,4,128,128,2
    concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout,stride,padding = float32,4,3,16,256,256,258,258,4,4,128,128,2,1

    # def add_padding(x,padding):
    #     return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    padding_input = np.zeros((batch_size,channel_in,height+2*padding,width+2*padding), dtype=np.float32)
    def kernel_convolution_layer[Ty, batch, cin, cout, height,width,heigth_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout,](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cout,cin, kernel_height, kernel_width]", Bias: "Ty[cout]",out:"Ty[batch, cout,hout,wout]"
            ):

        padded_input:Ty[batch, cin,heigth_after_padding,width_after_padding] = padding_input
        buffer_out: Ty[cout] = 0
        a: Ty = 0
        temp: Ty = 0
        for i, m, n, l in dsl.grid(batch, cin, height, width):
            padded_input[i, m, n + padding, l + padding] = input[i, m, n, l]

        for i, k, z, n, l in dsl.grid(batch, hout, wout, height_after_padding, width_after_padding):
            if n == k * stride and l == z * stride:  # Ensure proper stride-based indexing
                for j1 in range(cout):
                    buffer_out[j1] = Bias[j1]  # Initialize with bias                
                for m in range(cin):  # Iterate over input channels
                    for p in range(kernel_height):
                        for q in range(kernel_width,name="inner_loop"):
                            a = padded_input[i, m, n + p, l + q]
                            for j2 in range(cout,name="inner_computer"):  # Bounds check
                               buffer_out[j2] += Weight[j2, m, p, q] * a
                for j3 in range(cout):
                    out[i, j3, k, z] = buffer_out[j3]
    
    s_convolution = allo.customize(kernel_convolution_layer, instantiate=[concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout])
    s_convolution.unroll("j1",factor=4)
    s_convolution.unroll("j2",factor=4)
    s_convolution.unroll("j3",factor=4)

    s_convolution.partition(s_convolution.buffer_out, partition_type=2, dim=1, factor=4)
    s_convolution.partition(s_convolution.Weight, partition_type=2, dim=1, factor=4)
    s_convolution.partition(s_convolution.Bias, partition_type=2, dim=1, factor=4)
    s_convolution.partition(s_convolution.out, partition_type=2, dim=2, factor=4)
    # s_convolution.reshape(s_convolution.buffer_out, (8,2))
    s_convolution.pipeline("j1")
    s_convolution.pipeline("j2")
    s_convolution.pipeline("j3")
    def top[Ty, batch, cin, cout, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cout, cin,kernel_height, kernel_width]", Bias: "Ty[cout]", out:"Ty[batch, cout,hout,wout]"
    ):
        kernel_convolution_layer[float32,batch_size,channel_in,channel_out,height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout](input, Weight,Bias,out)


    s = allo.customize(top, instantiate=[concrete_type,  batch_size, channel_in, channel_out, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout])
    s.compose(s_convolution)

    return s.build(target="vitis_hls", mode="csyn", project="convolution_pipeline_unroll_2_reshape.prj")

# mod = Convolution_layer_unoptimized()
# mod()
# %%

# %%    
def test_convolution_layer():
    # Parameters for small test size
    # concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,height_out,widthout,stride,padding = int32,16,3,128,258,258,4,4,128,128,2,1
    concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,stride,padding = float32,1,32,16,256,256,4,4,2,1
    expanded_height = (height - 1) * stride + 1
    expanded_width = (width - 1) * stride + 1
    output_height = (height - 1) * stride + 1  + kernel_height - 1
    output_width = (width - 1) * stride + 1  + kernel_width - 1
    output_height_after_padding = output_height-2*padding
    output_width_after_padding = output_width-2*padding
    total_time = 0
    # Random input, weight, and bias initialization
    X = np.random.randn(batch_size, channel_in,height,width).astype(np.float32)
    W = np.random.randn(channel_in, channel_out,kernel_height,kernel_width).astype(np.float32)
    B = np.random.randn(channel_out).astype(np.float32)
    allo_C = np.zeros((batch_size, channel_out,output_height_after_padding,output_width_after_padding), dtype=np.float32)
     
    # Instantiate the layer using systolic optimizations
    mod = Convolution_transpose_unoptimized()
    print("finish compilation")
    print("Finish compilation, checking mod type:", type(mod))
    start_time = time.time()
    mod(X, W,B,allo_C)
    end_time = time.time()
    total_time += (end_time - start_time)
    print(f"Total time : {total_time:.5f} seconds")
    # ref = numpy_linear_layer(X, W, B)
    Convolution_layer = nn.ConvTranspose2d(32, 16,4,2,1)
    with torch.no_grad():  # Avoid tracking gradients during this operation
        Convolution_layer.weight.copy_(torch.from_numpy(W))
        Convolution_layer.bias.copy_(torch.from_numpy(B))
    
    ref = Convolution_layer(torch.from_numpy(X)).detach().numpy()
    np.savetxt("comparison.csv", np.c_[allo_C.flatten(), ref.flatten()], fmt="%.6f", delimiter=",", header="allo_C, ref", comments="")
    # Verify with NumPy reference
    # Verify with NumPy reference
    print("start comparing")
    np.testing.assert_allclose(allo_C, ref, rtol=1e-05,atol=1e-3)
    print("Test Passed!")


test_convolution_layer()

# %%