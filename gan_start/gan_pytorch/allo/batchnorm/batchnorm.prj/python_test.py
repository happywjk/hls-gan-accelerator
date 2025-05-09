import numpy as np
import torch
from torch import nn

def test_batchnorm_layer():
    # Parameters for small test size
    batch_size, num_channels, height, width = 4, 16, 64, 64
    
    # Random input initialization
    X = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
    
    # BatchNorm parameters
    gamma = np.random.randn(num_channels).astype(np.float32)  # Scale parameter
    beta = np.random.randn(num_channels).astype(np.float32)    # Shift parameter
    running_mean = np.zeros(num_channels, dtype=np.float32)    # Running mean (initially zeros)
    running_var = np.ones(num_channels, dtype=np.float32)      # Running variance (initially ones)
    
    # Create PyTorch BatchNorm for reference
    bn_layer = nn.BatchNorm2d(num_channels)
    with torch.no_grad():  # Avoid tracking gradients
        bn_layer.weight.copy_(torch.from_numpy(gamma))
        bn_layer.bias.copy_(torch.from_numpy(beta))
        bn_layer.running_mean.copy_(torch.from_numpy(running_mean))
        bn_layer.running_var.copy_(torch.from_numpy(running_var))
        bn_layer.eval()  # Set to evaluation mode
    
    # Get reference output from PyTorch
    ref = bn_layer(torch.from_numpy(X)).detach().numpy()
    
    # Flatten arrays in the correct order for C++ implementation
    X_flat = X.flatten().astype(np.float32)
    
    # Step 1: Write input data to files for host.cpp
    with open("input0.data", "w") as f:
        for val in X_flat:
            f.write(f"{val}\n")

    with open("input1.data", "w") as f:
        for val in gamma:
            f.write(f"{val}\n")

    with open("input2.data", "w") as f:
        for val in beta:
            f.write(f"{val}\n")
            
    with open("input3.data", "w") as f:
        for val in running_mean:
            f.write(f"{val}\n")
            
    with open("input4.data", "w") as f:
        for val in running_var:
            f.write(f"{val}\n")
            
    # Initialize output buffer with zeros
    output_size = batch_size * num_channels * height * width
    with open("input5.data", "w") as f:
        for _ in range(output_size):
            f.write("0.0\n")

    print("Please run host.cpp now to generate output.data, then press Enter to continue...")
    input()

    # Read and process the output
    output_data = []
    with open("output.data", "r") as f:
        for line in f:
            output_data.append(float(line.strip()))
    
    # Reshape to match expected dimensions
    output_array = np.array(output_data).reshape(batch_size, num_channels, height, width)

    # Compare with PyTorch reference
    print("Comparing with PyTorch reference...")
    try:
        np.testing.assert_allclose(output_array, ref, rtol=1e-05, atol=1e-03)
        print("Test PASSED! The tile-based BatchNorm matches the PyTorch reference.")
    except AssertionError as e:
        print("Test FAILED!")
        print(e)
    
    # Print some stats about the difference
    diff = np.abs(output_array - ref)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")

if __name__ == "__main__":
    test_batchnorm_layer()