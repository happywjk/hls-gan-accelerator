import numpy as np
import allo
from allo.ir.types import int32,float32
from allo import dsl
from torch import nn
import torch



def BatchNorm2d_layer_unoptimized():
    # Define concrete types and dimensions
    concrete_type, batch_size, num_channels, height, width = float32, 4, 16, 64, 64
    eps = 1e-5  # Small constant for numerical stability

    def kernel_batchnorm_layer[Ty, batch, channels, height, width](
            input: "Ty[batch, channels, height, width]",
            gamma: "Ty[channels]",  # Scale parameter
            beta: "Ty[channels]",   # Shift parameter
            running_mean: "Ty[channels]",  # Stored mean from training
            running_var: "Ty[channels]",   # Stored variance from training
            out: "Ty[batch, channels, height, width]"
            ):
        # For inference, we use the running statistics
        for n, c, h, w in dsl.grid(batch, channels, height, width):
            # Combine normalization, scale, and shift in a single expression
            out[n, c, h, w] = gamma[c] * ((input[n, c, h, w] - running_mean[c]) / dsl.sqrt(running_var[c] + eps)) + beta[c]
    
    s_batchnorm = allo.customize(kernel_batchnorm_layer, 
                                instantiate=[concrete_type, batch_size, num_channels, height, width])

    def top[Ty, batch, channels, height, width](
            input: "Ty[batch, channels, height, width]",
            gamma: "Ty[channels]",
            beta: "Ty[channels]",
            running_mean: "Ty[channels]",
            running_var: "Ty[channels]",
            out: "Ty[batch, channels, height, width]"
    ):
        kernel_batchnorm_layer[concrete_type, batch_size, num_channels, height, width](
            input, gamma, beta, running_mean, running_var, out)

    s = allo.customize(top, instantiate=[concrete_type, batch_size, num_channels, height, width])
    s.compose(s_batchnorm)

    return s.build(target="vitis_hls", mode="sw_emu", project="batchnorm.prj")

def test_batchnorm_layer():
    # Parameters for small test size
    concrete_type, batch_size, num_channels, height, width = float32, 4, 16, 64, 64
    total_time = 0
    
    # Random input initialization
    X = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
    
    # BatchNorm parameters
    gamma = np.random.randn(num_channels).astype(np.float32)  # Scale parameter
    beta = np.random.randn(num_channels).astype(np.float32)    # Shift parameter
    running_mean = np.zeros(num_channels, dtype=np.float32)    # Running mean (initially zeros)
    running_var = np.ones(num_channels, dtype=np.float32)      # Running variance (initially ones)
    
    # Output buffer
    allo_output = np.zeros((batch_size, num_channels, height, width), dtype=np.float32)
    
    # Instantiate the BatchNorm layer
    mod = BatchNorm2d_layer_unoptimized()
    print("finish compilation")
    print("Finish compilation, checking mod type:", type(mod))
    
    # Test in inference mode (is_training = False)
    

    mod(X, gamma, beta, running_mean, running_var, allo_output)


    print(f"Total time : {total_time:.5f} seconds")
    
    # Create PyTorch BatchNorm for reference
    bn_layer = nn.BatchNorm2d(num_channels)
    with torch.no_grad():  # Avoid tracking gradients
        bn_layer.weight.copy_(torch.from_numpy(gamma))
        bn_layer.bias.copy_(torch.from_numpy(beta))
        bn_layer.running_mean.copy_(torch.from_numpy(running_mean))
        bn_layer.running_var.copy_(torch.from_numpy(running_var))
        bn_layer.eval()  # Set to evaluation mode (equivalent to is_training=False)
    
    # Get reference output from PyTorch
    ref = bn_layer(torch.from_numpy(X)).detach().numpy()
    
    # Verify with PyTorch reference
    print("start comparing")
    np.testing.assert_allclose(allo_output, ref, rtol=1e-05, atol=1e-3)
    print("Test Passed!")

test_batchnorm_layer()