import numpy as np
import torch
from torch import nn

def test_relu_layer_fileio():
    # Parameters for small test size
    batch_size, num_channels, height, width = 4, 16, 64, 64
    
    # Random input initialization (including negative values to test ReLU)
    X = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
    
    # Create PyTorch ReLU for reference
    relu_layer = nn.ReLU()
    
    # Get reference output from PyTorch
    ref = relu_layer(torch.from_numpy(X)).detach().numpy()
    ref_1 = relu_layer(torch.from_numpy(ref)).detach().numpy()
    
    # Step 1: Write input data to files for host.cpp
    # input0.data: X flattened (4 * 16 * 64 * 64 = 262,144 floats)
    with open("input0.data", "w") as f:
        for val in X.flatten():
            f.write(f"{val}\n")

    # input1.data: Initialize with zeros for output buffer (262,144 floats)
    with open("input1.data", "w") as f:
        for _ in range(batch_size * num_channels * height * width):
            f.write("0.0\n")

    print("Input files written:")
    print("  input0.data - Input tensor X")
    print("  input1.data - Output buffer (zeros)")
    print("Please run host.cpp now to generate output.data, then press Enter to continue...")
    input()  # Wait for user to run host.cpp

    # Step 2: Read output.data produced by host.cpp
    with open("output.data", "r") as f:
        output_data = [float(line.strip()) for line in f]

    # Step 3: Reshape output.data to match expected dimensions
    allo_output = np.array(output_data).reshape(batch_size, num_channels, height, width)

    # Step 4: Compare with the reference output
    print("Comparing output from host.cpp with PyTorch reference output...")
    np.testing.assert_allclose(allo_output, ref, rtol=1e-05, atol=1e-03)
    print("Test Passed! The C++ implementation matches the PyTorch reference.")
    
    # For additional verification
    negative_inputs = np.sum(X < 0)
    zero_outputs = np.sum(allo_output == 0)
    print(f"Number of negative inputs: {negative_inputs}")
    print(f"Number of zero outputs: {zero_outputs}")
    print(f"These should be equal for a correct ReLU implementation")

if __name__ == "__main__":
    test_relu_layer_fileio()