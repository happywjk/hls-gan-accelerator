# %%

import numpy as np
import allo
from allo.ir.types import int32,float32
from allo import dsl
from torch import nn
import torch
import time
from allo.ir.utils import MockBuffer
syn_config = {
    "compile.pipeline_loops": False  # Disables loop pipelining
}



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
for i,j in np.ndindex(2, 3):
    print(i,j)
# %%
# unoptimized version
def Convolution_layer_unoptimized():
    # concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,height_out,widthout,stride = int32,16,3,128,258,258,4,4,128,128,2
    concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout,stride,padding = float32,4,3,16,256,256,258,258,4,4,128,128,2,1

    # def add_padding(x,padding):
    #     return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    def kernel_convolution_layer[Ty, batch, cin, cout, height,width,heigth_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout,](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cout,cin, kernel_height, kernel_width]", Bias: "Ty[cout]",out:"Ty[batch, cout,hout,wout]"
            ):
        padded_input:Ty[batch, cin,heigth_after_padding,width_after_padding] = 0
        for i, m, n, l in dsl.grid(batch, cin, height, width):
            padded_input[i, m, n + padding, l + padding] = input[i, m, n, l]

        for i, j, k, z, n, l in dsl.grid(batch, cout, hout, wout, height_after_padding, width_after_padding):
            if n == k * stride and l == z * stride:  # Ensure proper stride-based indexing
                out[i, j, k, z] = Bias[j]  # Initialize with bias                
                for m in range(cin):  # Iterate over input channels
                    for p in range(kernel_height):
                        for q in range(kernel_width,name="inner_loop"):
                            if n + p < height_after_padding and l + q < width_after_padding:  # Bounds check
                                out[i, j, k, z] += Weight[j, m, p, q] * padded_input[i, m, n + p, l + q]
    
    s_linear = allo.customize(kernel_convolution_layer, instantiate=[concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout])

    def top[Ty, batch, cin, cout, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cout, cin,kernel_height, kernel_width]", Bias: "Ty[cout]", out:"Ty[batch, cout,hout,wout]"
    ):
        kernel_convolution_layer[float32,batch_size,channel_in,channel_out,height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout](input, Weight,Bias,out)

    s = allo.customize(top, instantiate=[concrete_type,  batch_size, channel_in, channel_out, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout])
    s.compose(s_linear)

    return s.build(target="vitis_hls", mode="csyn", project="convolution_unopitimized.prj")

# mod = Convolution_layer_unoptimized()
# mod()

# %%
# pipeline version
def Convolution_layer_pipeline():
    # concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,height_out,widthout,stride = int32,16,3,128,258,258,4,4,128,128,2
    concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout,stride,padding = float32,4,3,16,256,256,258,258,4,4,128,128,2,1

    # def add_padding(x,padding):
    #     return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    padding_input = np.zeros((batch_size,channel_in,height+2*padding,width+2*padding), dtype=np.float32)
    def kernel_convolution_layer[Ty, batch, cin, cout, height,width,heigth_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout](
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
        kernel_convolution_layer()

    s = allo.customize(top, instantiate=[concrete_type,  batch_size, channel_in, channel_out, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout])
    s.compose(s_convolution)

    return s.build(target="vitis_hls", mode="csyn", project="convolution_pipeline_unroll_4.prj")

# mod = Convolution_layer_unoptimized()
# mod()
# %%

# %%    
def test_convolution_layer():
    # Parameters for small test size
    # concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,height_out,widthout,stride,padding = int32,16,3,128,258,258,4,4,128,128,2,1
    concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout,stride,padding = float32,4,3,16,256,256,258,258,4,4,128,128,2,1
    total_time = 0
    # Random input, weight, and bias initialization
    X = np.random.randn(batch_size, channel_in,height,width).astype(np.float32)
    W = np.random.randn(channel_out, channel_in,kernel_height,kernel_width).astype(np.float32)
    B = np.random.randn(channel_out).astype(np.float32)
    allo_C = np.zeros((batch_size, channel_out,height_out,widthout), dtype=np.float32)
     
    # Instantiate the layer using systolic optimizations
    mod = Convolution_layer_unoptimized()
    print("finish compilation")
    print("Finish compilation, checking mod type:", type(mod))
    start_time = time.time()
    mod(X, W,B,allo_C)
    end_time = time.time()
    total_time += (end_time - start_time)
    print(f"Total time : {total_time:.5f} seconds")
    # ref = numpy_linear_layer(X, W, B)
    Convolution_layer = nn.Conv2d(3, 16,4,2,1)
    with torch.no_grad():  # Avoid tracking gradients during this operation
        Convolution_layer.weight.copy_(torch.from_numpy(W))
        Convolution_layer.bias.copy_(torch.from_numpy(B))
    
    ref = Convolution_layer(torch.from_numpy(X)).detach().numpy()
    # Verify with NumPy reference
    # Verify with NumPy reference
    print("start comparing")
    np.testing.assert_allclose(allo_C, ref, rtol=1e-05,atol=1e-3)
    print("Test Passed!")


test_convolution_layer()

# %%
