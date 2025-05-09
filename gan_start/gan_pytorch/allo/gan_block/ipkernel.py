import numpy as np
import torch
from torch import nn

def test_genblock():
    # Parameters for the test
    batch_size, channel_in, channel_out = 4, 3, 16
    height, width = 64, 64
    kernel_height, kernel_width = 4, 4
    stride, padding = 2, 0

    # Compute output dimensions
    height_out = stride * (height - 1) + kernel_height - 2 * padding
    width_out = stride * (width - 1) + kernel_width - 2 * padding

    # Random input, weight, and bias initialization
    X = np.random.randn(batch_size, channel_in, height, width).astype(np.float32)
    W = np.random.randn(channel_in, channel_out, kernel_height, kernel_width).astype(np.float32)
    B = np.random.randn(channel_out).astype(np.float32)
    
    # Random BatchNorm parameters
    gamma = np.random.randn(channel_out).astype(np.float32)
    beta = np.random.randn(channel_out).astype(np.float32)
    running_mean = np.random.randn(channel_out).astype(np.float32)
    running_var = np.random.rand(channel_out).astype(np.float32) + 0.1  # Make sure variance is positive
    
    # Define PyTorch GenBlock
    class GenBlock(nn.Module):
        def __init__(self, in_channels, out_channels, padding):
            super(GenBlock, self).__init__()
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 4, 2, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Create GenBlock and set parameters
    gen_block = GenBlock(channel_in, channel_out, padding)
    gen_block.eval()
    # Set the weights and parameters
    with torch.no_grad():
        # Set ConvTranspose2d weights and biases
        gen_block.layers[0].weight.copy_(torch.from_numpy(W))
        gen_block.layers[0].bias.copy_(torch.from_numpy(B))
        
        # Set BatchNorm2d parameters
        gen_block.layers[1].weight.copy_(torch.from_numpy(gamma))  # gamma
        gen_block.layers[1].bias.copy_(torch.from_numpy(beta))     # beta
        gen_block.layers[1].running_mean.copy_(torch.from_numpy(running_mean))
        gen_block.layers[1].running_var.copy_(torch.from_numpy(running_var))

    # Compute reference output using PyTorch
    ref = gen_block(torch.from_numpy(X)).detach().numpy()

    # Write input data to files for host.cpp
    # input0.data: X (input tensor)
    with open("input0.data", "w") as f:
        for val in X.flatten():
            f.write(f"{val}\n")

    # input1.data: W (convolution weights)
    with open("input1.data", "w") as f:
        for val in W.flatten():
            f.write(f"{val}\n")

    # input2.data: B (convolution bias)
    with open("input2.data", "w") as f:
        for val in B:
            f.write(f"{val}\n")
    
    # input3.data: gamma (BatchNorm weight)
    with open("input3.data", "w") as f:
        for val in gamma:
            f.write(f"{val}\n")
    
    # input4.data: beta (BatchNorm bias)
    with open("input4.data", "w") as f:
        for val in beta:
            f.write(f"{val}\n")
    
    # input5.data: running_mean
    with open("input5.data", "w") as f:
        for val in running_mean:
            f.write(f"{val}\n")
    
    # input6.data: running_var
    with open("input6.data", "w") as f:
        for val in running_var:
            f.write(f"{val}\n")

    # input7.data: Initialize output buffer with zeros
    with open("input7.data", "w") as f:
        output_size = batch_size * channel_out * height_out * width_out
        for _ in range(output_size):
            f.write("0.0\n")

    print("Input files written: input0.data through input7.data")
    print("Please run host.cpp now to generate output.data, then press Enter to continue...")
    input()  # Wait for user to run host.cpp

    # Read output.data produced by host.cpp
    with open("output.data", "r") as f:
        output_data = [float(line.strip()) for line in f]

    # Reshape output.data to match expected dimensions
    allo_C = np.array(output_data).reshape(batch_size, channel_out, height_out, width_out)

    # Compare with the reference output
    print("Comparing output from host.cpp with reference output...")
    np.testing.assert_allclose(allo_C, ref, rtol=1e-05, atol=1e-03)
    print("Test Passed! GenBlock implementation works correctly.")

if __name__ == "__main__":
    test_genblock()