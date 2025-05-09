import numpy as np
import torch
from torch import nn

def test_generator():
    # Parameters for the test
    batch_size = 8
    z_dim = 128  # Input channels for the Generator
    
    # Initial dimensions for latent vector
    init_height, init_width = 1, 1
    
    # Random input initialization (latent vector z)
    z = np.random.randn(batch_size, z_dim, init_height, init_width).astype(np.float32)
    
    # Define Generator class (already provided by the user, including GenBlock)
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
    
    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.layers = nn.Sequential(
                GenBlock(128, 256, 0),     # 1x1 -> 4x4
                GenBlock(256, 128, 1),     # 4x4 -> 8x8
                GenBlock(128, 64, 1),      # 8x8 -> 16x16
                GenBlock(64, 32, 1),      # 16x16 -> 32x32
                GenBlock(32, 16, 1),       # 32x32 -> 64x64
                nn.ConvTranspose2d(16, 3, 4, 2, 1), # 64x64 -> 128x128
                nn.Tanh()
            )
    
        def forward(self, x):
            return self.layers(x)
    
    # Create Generator and set to evaluation mode
    generator = Generator()
    generator.eval()
    
    # Layer parameters
    layer_channels = [(128, 256), (256, 128), (128, 64), (64, 32), (32, 16), (16, 3)]
    kernel_size = 4
    
    # Lists to store parameters
    weights = []
    biases = []
    bn_weights = []
    bn_biases = []
    bn_means = []
    bn_vars = []
    
    # Initialize parameters with random values
    for i, (in_channels, out_channels) in enumerate(layer_channels):
        # ConvTranspose2d weights and biases
        W = np.random.randn(in_channels, out_channels, kernel_size, kernel_size).astype(np.float32)
        B = np.random.randn(out_channels).astype(np.float32)
        weights.append(W)
        biases.append(B)
        
        # For GenBlocks (all but the last layer), add BatchNorm parameters
        if i < len(layer_channels) - 1:  # Not adding BatchNorm for the final layer
            gamma = np.random.randn(out_channels).astype(np.float32)
            beta = np.random.randn(out_channels).astype(np.float32)
            mean = np.random.randn(out_channels).astype(np.float32)
            var = np.random.rand(out_channels).astype(np.float32) + 0.1  # Make sure variance is positive
            
            bn_weights.append(gamma)
            bn_biases.append(beta)
            bn_means.append(mean)
            bn_vars.append(var)
    
    # Set the weights and parameters to the Generator model
    with torch.no_grad():
        # Set parameters for each GenBlock
        for i in range(5):  # 5 GenBlocks
            # Set ConvTranspose2d weights and biases
            generator.layers[i].layers[0].weight.copy_(torch.from_numpy(weights[i]))
            generator.layers[i].layers[0].bias.copy_(torch.from_numpy(biases[i]))
            
            # Set BatchNorm2d parameters
            generator.layers[i].layers[1].weight.copy_(torch.from_numpy(bn_weights[i]))
            generator.layers[i].layers[1].bias.copy_(torch.from_numpy(bn_biases[i]))
            generator.layers[i].layers[1].running_mean.copy_(torch.from_numpy(bn_means[i]))
            generator.layers[i].layers[1].running_var.copy_(torch.from_numpy(bn_vars[i]))
        
        # Set final ConvTranspose2d weights and biases
        generator.layers[5].weight.copy_(torch.from_numpy(weights[5]))
        generator.layers[5].bias.copy_(torch.from_numpy(biases[5]))
    
    # Compute reference output using PyTorch
    ref = generator(torch.from_numpy(z)).detach().numpy()
    
    # Expected output dimensions
    final_height = 128  # As per comments
    final_width = 128
    final_channels = 3  # RGB output
    
    # Write input data to files for host.cpp
    # input0.data: z (input tensor)
    with open("input0.data", "w") as f:
        for val in z.flatten():
            f.write(f"{val}\n")
    
    # Write all weights, biases, and BatchNorm parameters
    file_idx = 1
    for i in range(len(weights)):
        # Write weights for ConvTranspose2d
        with open(f"input{file_idx}.data", "w") as f:
            for val in weights[i].flatten():
                f.write(f"{val}\n")
        file_idx += 1
        
        # Write biases for ConvTranspose2d
        with open(f"input{file_idx}.data", "w") as f:
            for val in biases[i]:
                f.write(f"{val}\n")
        file_idx += 1
        
        # For GenBlocks, write BatchNorm parameters
        if i < len(weights) - 1:  # All but the last layer
            # Write gamma (BatchNorm weight)
            with open(f"input{file_idx}.data", "w") as f:
                for val in bn_weights[i]:
                    f.write(f"{val}\n")
            file_idx += 1
            
            # Write beta (BatchNorm bias)
            with open(f"input{file_idx}.data", "w") as f:
                for val in bn_biases[i]:
                    f.write(f"{val}\n")
            file_idx += 1
            
            # Write running_mean
            with open(f"input{file_idx}.data", "w") as f:
                for val in bn_means[i]:
                    f.write(f"{val}\n")
            file_idx += 1
            
            # Write running_var
            with open(f"input{file_idx}.data", "w") as f:
                for val in bn_vars[i]:
                    f.write(f"{val}\n")
            file_idx += 1
    
    # Initialize output buffer with zeros
    with open(f"input{file_idx}.data", "w") as f:
        output_size = batch_size * final_channels * final_height * final_width
        for _ in range(output_size):
            f.write("0.0\n")
    
    print(f"Input files written: input0.data through input{file_idx}.data")
    print("Please run host.cpp now to generate output.data, then press Enter to continue...")
    input()  # Wait for user to run host.cpp
    
    # Read output.data produced by host.cpp
    with open("output.data", "r") as f:
        output_data = [float(line.strip()) for line in f]
    
    # Reshape output.data to match expected dimensions
    allo_C = np.array(output_data).reshape(batch_size, final_channels, final_height, final_width)
    
    # Compare with the reference output
    print("Comparing output from host.cpp with reference output...")
    np.testing.assert_allclose(allo_C, ref, rtol=1e-05, atol=1e-03)
    print("Test Passed! Generator implementation works correctly.")

if __name__ == "__main__":
    test_generator()