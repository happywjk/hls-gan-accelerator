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
    
    # Lists to store parameters - organize them in the order matching the first two scripts
    all_weights = []
    all_biases = []
    all_bn_weights = []
    all_bn_biases = []
    all_bn_means = []
    all_bn_vars = []
    
    # Initialize parameters with random values
    for i, (in_channels, out_channels) in enumerate(layer_channels):
        # ConvTranspose2d weights
        W = np.random.randn(in_channels, out_channels, kernel_size, kernel_size).astype(np.float32)
        all_weights.append(W)
        
        # For the first 5 layers (GenBlocks), create zero biases as in the first script
        # For the final layer, use random biases
        if i < len(layer_channels) - 1:  # For GenBlocks
            B = np.zeros(out_channels).astype(np.float32)  # Zero biases for GenBlocks
            
            # BatchNorm parameters
            gamma = np.random.randn(out_channels).astype(np.float32)
            beta = np.random.randn(out_channels).astype(np.float32)
            mean = np.random.randn(out_channels).astype(np.float32)
            # Make sure variance is positive and reasonably valued
            var = np.random.rand(out_channels).astype(np.float32) * 0.1 + 1.0
            
            all_bn_weights.append(gamma)
            all_bn_biases.append(beta)
            all_bn_means.append(mean)
            all_bn_vars.append(var)
        else:  # For the final ConvTranspose2d layer
            B = np.random.randn(out_channels).astype(np.float32)
        
        all_biases.append(B)
    
    # Set the weights and parameters to the Generator model
    with torch.no_grad():
        # Set parameters for each GenBlock
        for i in range(5):  # 5 GenBlocks
            # Set ConvTranspose2d weights and biases
            generator.layers[i].layers[0].weight.copy_(torch.from_numpy(all_weights[i]))
            generator.layers[i].layers[0].bias.copy_(torch.from_numpy(all_biases[i]))
            
            # Set BatchNorm2d parameters
            generator.layers[i].layers[1].weight.copy_(torch.from_numpy(all_bn_weights[i]))
            generator.layers[i].layers[1].bias.copy_(torch.from_numpy(all_bn_biases[i]))
            generator.layers[i].layers[1].running_mean.copy_(torch.from_numpy(all_bn_means[i]))
            generator.layers[i].layers[1].running_var.copy_(torch.from_numpy(all_bn_vars[i]))
        
        # Set final ConvTranspose2d weights and biases
        generator.layers[5].weight.copy_(torch.from_numpy(all_weights[5]))
        generator.layers[5].bias.copy_(torch.from_numpy(all_biases[5]))
    
    # Compute reference output using PyTorch
    ref = generator(torch.from_numpy(z)).detach().numpy()
    
    # Expected output dimensions
    final_height = 128
    final_width = 128
    final_channels = 3  # RGB output
    
    # Write input data to file for host.cpp
    with open("input0.data", "w") as f:
        for val in z.flatten():
            f.write(f"{val}\n")
    
    # Now write files in correct order following file_mapping pattern from first two scripts
    
    # 1. Write all weights (Input1-6)
    for i in range(len(all_weights)):
        with open(f"input{i+1}.data", "w") as f:
            for val in all_weights[i].flatten():
                f.write(f"{val}\n")
    
    # 2. Write all biases (Input7-12)
    bias_start_idx = len(all_weights) + 1
    for i in range(len(all_biases)):
        with open(f"input{bias_start_idx + i}.data", "w") as f:
            for val in all_biases[i]:
                f.write(f"{val}\n")
    
    # 3. Write all BatchNorm weights (Input13-17)
    bn_weight_start_idx = bias_start_idx + len(all_biases)
    for i in range(len(all_bn_weights)):
        with open(f"input{bn_weight_start_idx + i}.data", "w") as f:
            for val in all_bn_weights[i]:
                f.write(f"{val}\n")
    
    # 4. Write all BatchNorm biases (Input18-22)
    bn_bias_start_idx = bn_weight_start_idx + len(all_bn_weights)
    for i in range(len(all_bn_biases)):
        with open(f"input{bn_bias_start_idx + i}.data", "w") as f:
            for val in all_bn_biases[i]:
                f.write(f"{val}\n")
    
    # 5. Write all BatchNorm running means (Input23-27)
    bn_mean_start_idx = bn_bias_start_idx + len(all_bn_biases)
    for i in range(len(all_bn_means)):
        with open(f"input{bn_mean_start_idx + i}.data", "w") as f:
            for val in all_bn_means[i]:
                f.write(f"{val}\n")
    
    # 6. Write all BatchNorm running vars (Input28-32)
    bn_var_start_idx = bn_mean_start_idx + len(all_bn_means)
    for i in range(len(all_bn_vars)):
        with open(f"input{bn_var_start_idx + i}.data", "w") as f:
            for val in all_bn_vars[i]:
                f.write(f"{val}\n")
    
    # Initialize output buffer with zeros (for host.cpp to write to)
    output_buffer_idx = bn_var_start_idx + len(all_bn_vars)
    with open(f"input{output_buffer_idx}.data", "w") as f:
        output_size = batch_size * final_channels * final_height * final_width
        for _ in range(output_size):
            f.write("0.0\n")
    
    # Save the expected output to a file for verification
    with open("expected_output.data", "w") as f:
        for val in ref.flatten():
            f.write(f"{val}\n")
    
    print(f"Input files written: input0.data through input{output_buffer_idx}.data")
    print("Please run host.cpp now to generate output.data, then press Enter to continue...")
    input()  # Wait for user to run host.cpp
    
    # Read output.data produced by host.cpp
    try:
        with open("output.data", "r") as f:
            output_data = [float(line.strip()) for line in f]
        
        # Reshape output.data to match expected dimensions
        allo_C = np.array(output_data).reshape(batch_size, final_channels, final_height, final_width)
        
        # Compare with the reference output
        print("Comparing output from host.cpp with reference output...")
        np.testing.assert_allclose(allo_C, ref, rtol=1e-05, atol=1e-03)
        print("Test Passed! Generator implementation works correctly.")
    except FileNotFoundError:
        print("Output.data file not found. Did you run host.cpp?")
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_generator()