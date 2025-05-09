import numpy as np
import allo
from allo.ir.types import int32, float32
from allo import dsl
from torch import nn
import torch


def ReLU_layer_unoptimized():
    # Define concrete types and dimensions
    concrete_type, batch_size, num_channels, height, width = float32, 4, 16, 64, 64

    def kernel_relu_layer[Ty, batch, channels, height, width](
            input: "Ty[batch, channels, height, width]",
            out: "Ty[batch, channels, height, width]"
            ):
        # Implement ReLU using (x + abs(x)) / 2 formula without temporary variables
        for n, c, h, w in dsl.grid(batch, channels, height, width):
            # Apply the ReLU formula directly without temporary variables
            if input[n,c,h,w] > 0:
                out[n, c, h, w] = input[n,c,h,w]
            else:
                out[n,c,h,w] = 0
    s_relu = allo.customize(kernel_relu_layer, 
                          instantiate=[concrete_type, batch_size, num_channels, height, width])

    def top[Ty, batch, channels, height, width](
            input: "Ty[batch, channels, height, width]",
            out: "Ty[batch, channels, height, width]"
    ):
        kernel_relu_layer[concrete_type, batch_size, num_channels, height, width](input, out)

    s = allo.customize(top, instantiate=[concrete_type, batch_size, num_channels, height, width])
    s.compose(s_relu)

    return s.build(target="vitis_hls", mode="sw_emu", project="relu.prj")


def test_relu_layer():
    # Parameters for small test size
    batch_size, num_channels, height, width = 4, 16, 64, 64
    
    # Random input initialization (including some negative values to test ReLU)
    X = np.random.randn(batch_size, num_channels, height, width).astype(np.float32)
    
    # Output buffer
    allo_output = np.zeros((batch_size, num_channels, height, width), dtype=np.float32)
    
    # Instantiate the ReLU layer
    mod = ReLU_layer_unoptimized()
    print("Finished compilation")
    print("Checking mod type:", type(mod))
    
    # Run ReLU function
    mod(X, allo_output)
    
    # Create PyTorch ReLU for reference
    relu_layer = nn.ReLU()
    
    # Get reference output from PyTorch
    ref = relu_layer(torch.from_numpy(X)).detach().numpy()
    
    # Verify with PyTorch reference
    print("Starting comparison")
    np.testing.assert_allclose(allo_output, ref, rtol=1e-05, atol=1e-3)
    print("Test Passed!")
    
    # For additional verification, do a manual check
    manual_relu = np.maximum(0, X)
    max_diff_manual = np.max(np.abs(allo_output - manual_relu))
    print(f"Maximum difference with manual ReLU implementation: {max_diff_manual}")
    
    # Print some statistics for verification
    negative_inputs = np.sum(X < 0)
    zero_outputs = np.sum(allo_output == 0)
    print(f"Number of negative inputs: {negative_inputs}")
    print(f"Number of zero outputs: {zero_outputs}")
    print(f"These should be equal for ReLU")


if __name__ == "__main__":
    test_relu_layer()