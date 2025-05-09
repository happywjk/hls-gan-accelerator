import numpy as np
import allo
from allo.ir.types import int32,float32
from allo import dsl
from torch import nn
import torch
import time

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

    Convolution_layer_unoptimized = allo.IPModule(
        top="top",
        impl="conv2d.cpp",
        link_hls=True,
    )


    # Instantiate the layer using systolic optimizations    
    start_time = time.time()
    Convolution_layer_unoptimized(X,W,B,allo_C)
    end_time = time.time()
    total_time += (end_time - start_time)
    print(f"Total time : {total_time:.5f} seconds")
    # ref = numpy_linear_layer(X, W, B)
    Convolution_layer = nn.Conv2d(3, 16,4,1,1)
    with torch.no_grad():  # Avoid tracking gradients during this operation
        Convolution_layer.weight.copy_(torch.from_numpy(W))
        Convolution_layer.bias.copy_(torch.from_numpy(B))
    
    ref = Convolution_layer(torch.from_numpy(X)).detach().numpy()

    print("start comparing")
    np.testing.assert_allclose(allo_C, ref, rtol=1e-05,atol=1e-3)
    print("Test Passed!")

test_convolution_layer()