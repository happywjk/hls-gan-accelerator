# import numpy as np
# import allo
# from allo.ir.types import int32,float32
# from allo import dsl
# from torch import nn
# import torch
# import time



import numpy as np
import torch
from torch import nn

def test_convolution_layer():
    # Parameters for the test
    batch_size, channel_in, channel_out = 8, 32, 16
    height, width = 32, 32
    kernel_height, kernel_width = 4, 4
    stride, padding = 2, 1

    # Compute output dimensions
    height_after_padding = height + 2 * padding
    width_after_padding = width + 2 * padding
    height_out = stride * (height -1) +kernel_height - 2*padding
    width_out = stride * (width -1) +kernel_width - 2*padding

    # Random input, weight, and bias initialization
    X = np.random.randn(batch_size, channel_in, height, width).astype(np.float32)
    W = np.random.randn(channel_in, channel_out, kernel_height, kernel_width).astype(np.float32)
    B = np.random.randn(channel_out).astype(np.float32)
    allo_C = np.zeros((batch_size, channel_out, height_out, width_out), dtype=np.float32)

    # Define PyTorch convolution layer and set weights and biases
    Convolution_transpose_layer = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=4, stride=stride, padding=padding)
    with torch.no_grad():
        Convolution_transpose_layer.weight.copy_(torch.from_numpy(W))
        Convolution_transpose_layer.bias.copy_(torch.from_numpy(B))

    # Compute reference output
    ref = Convolution_transpose_layer(torch.from_numpy(X)).detach().numpy()

    # Step 1: Write input data to files for host.cpp
    # input0.data: X flattened (4 * 3 * 64 * 64 = 49152 floats)
    with open("input0.data", "w") as f:
        for val in X.flatten():
            f.write(f"{val}\n")

    # input1.data: W flattened (16 * 3 * 4 * 4 = 768 floats)
    with open("input1.data", "w") as f:
        for val in W.flatten():
            f.write(f"{val}\n")

    # input2.data: B (16 floats)
    with open("input2.data", "w") as f:
        for val in B:
            f.write(f"{val}\n")

    # input3.data: Initialize with zeros (65536 floats, matching host.cpp output buffer size)
    with open("input3.data", "w") as f:
        for _ in range(28672):
            f.write("0.0\n")

    print("Input files written: input0.data, input1.data, input2.data, input3.data")
    print("Please run host.cpp now to generate output.data, then press Enter to continue...")
    input()  # Wait for user to run host.cpp

    # Step 2: Read output.data produced by host.cpp
    with open("output.data", "r") as f:
        output_data = [float(line.strip()) for line in f]

    # Step 3: Reshape output.data to (16, 64, 64) - assuming it represents one batch
    allo_C_part = np.array(output_data).reshape(batch_size,channel_out, height_out, width_out)

    # Step 4: Compare with the first batch of the reference output
    print("Comparing output from host.cpp with reference output for the first batch...")
    np.testing.assert_allclose(allo_C_part, ref, rtol=1e-05, atol=1e-03)
    print("Test Passed")

if __name__ == "__main__":
    test_convolution_layer()