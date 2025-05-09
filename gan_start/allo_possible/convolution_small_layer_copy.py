# %%

import numpy as np
import allo
from allo.ir.types import int32,float32
from allo import dsl
from torch import nn
import torch
import time
from allo.ir.utils import MockBuffer




# %%
# unoptimized version
def Convolution_layer_unoptimized_reuse():
    concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,stride,padding = float32,4,3,16,64,64,4,4,1,1
    height_after_padding, width_after_padding = height + 2*padding, width + 2*padding
    height_out, widthout = (height_after_padding - kernel_height) // stride + 1, (width_after_padding - kernel_width) // stride + 1

    # def add_padding(x,padding):
    #     return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    def kernel_convolution_layer[Ty, batch, cin, cout, height,width,heigth_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cout,cin, kernel_height, kernel_width]", Bias: "Ty[cout]",out:"Ty[batch, cout,hout,wout]"
            ):
        padded_input:Ty[batch, cin,heigth_after_padding,width_after_padding] = 0
        for i, m, n, l in dsl.grid(batch, cin, height, width):
            padded_input[i, m, n + padding, l + padding] = input[i, m, n, l]

        for i, j, k, z in dsl.grid(batch, cout, hout, wout):
                out[i, j, k, z] = Bias[j]  # Initialize with bias 
                # v:float32 = 0.0               
                for m,p,q in allo.reduction(cin,kernel_height,kernel_width):  # Iterate over input channels
                    out[i, j, k, z] += Weight[j, m, p, q] * padded_input[i, m, k*stride + p, z*stride + q]
                # out[i, j, k, z] = v
    
    s_conv = allo.customize(kernel_convolution_layer, instantiate=[concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout])
    LB= s_conv.reuse_at(s_conv.padded_input,"k")
    WB= s_conv.reuse_at(LB,"z")


    def top[Ty, batch, cin, cout, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cout, cin,kernel_height, kernel_width]", Bias: "Ty[cout]", out:"Ty[batch, cout,hout,wout]"
    ):
        kernel_convolution_layer[float32,batch_size,channel_in,channel_out,height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout](input, Weight,Bias,out)

    s = allo.customize(top, instantiate=[concrete_type,  batch_size, channel_in, channel_out, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout])
    s.compose(s_conv)

    return s.build(target="vitis_hls", mode="sw_emu", project="convolution_reuse.prj")

# mod = Convolution_layer_unoptimized()
# mod()

# %%

# %%    
def test_convolution_layer():
    # Parameters for small test size
    concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,stride,padding = float32,4,3,16,64,64,4,4,1,1
    height_after_padding, width_after_padding = height + 2*padding, width + 2*padding
    height_out, widthout = (height_after_padding - kernel_height) // stride + 1, (width_after_padding - kernel_width) // stride + 1
    total_time = 0
    # Random input, weight, and bias initialization
    X = np.random.randn(batch_size, channel_in,height,width).astype(np.float32)
    W = np.random.randn(channel_out, channel_in,kernel_height,kernel_width).astype(np.float32)
    B = np.random.randn(channel_out).astype(np.float32)
    allo_C = np.zeros((batch_size, channel_out,height_out,widthout), dtype=np.float32)
     
    # Instantiate the layer using systolic optimizations
    mod = Convolution_layer_unoptimized_reuse()
    print("finish compilation")
    print("Finish compilation, checking mod type:", type(mod))
    start_time = time.time()
    mod(X, W,B,allo_C)
    end_time = time.time()
    total_time += (end_time - start_time)
    print(f"Total time : {total_time:.5f} seconds")
    # ref = numpy_linear_layer(X, W, B)
    Convolution_layer = nn.Conv2d(3, 16,4,1,1)
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
