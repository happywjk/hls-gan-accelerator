# %%

import numpy as np
import allo
from allo.ir.types import int32,float32
from allo import dsl
from torch import nn
import torch
from allo.ir.utils import MockBuffer


# %%


for i,j in np.ndindex(2, 3):
    print(i,j)
# %%
# unoptimized version
def linear_layer_unoptimized():
    concrete_type, L, D, O = float32, 1024, 1024, 1024
    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", W: "Ty[D, O]", B: "Ty[O]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        for i, j in dsl.grid(L, O):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D):
                Z[i, j] += X[i, k] * W[k, j]
    

    s_linear = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])

    def top[Ty, L, D, O](a: "Ty[L, D]",W: "Ty[D, O]", B: "Ty[O]", b: "Ty[L, O]"):
        kernel_linear_layer[float32, 1024, 1024, 1024](a, W,B, b)

    s = allo.customize(top, instantiate=[concrete_type, L, D, O])
    s.compose(s_linear)

    return s.build(target="vitis_hls", mode="csyn", project="unoptimized.prj")

# mod = linear_layer_unoptimized()
# mod()
# %%
# pipelined version
def linear_layer_pipelined():
    concrete_type, L, D, O = float32, 1024, 1024, 1024
    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", W: "Ty[D, O]", B: "Ty[O]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        for i, j in dsl.grid(L, O):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D):
                Z[i, j] += X[i, k] * W[k, j]
    

    s_linear = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])
    s_linear.pipeline("k")

    def top[Ty, L, D, O](a: "Ty[L, D]",W: "Ty[D, O]", B: "Ty[O]", b: "Ty[L, O]"):
        kernel_linear_layer[float32, 1024, 1024, 1024](a, W,B, b)

    s = allo.customize(top, instantiate=[concrete_type, L, D, O])
    s.compose(s_linear)

    return s.build(target="vitis_hls", mode="csyn", project="pipelined.prj")

mod = linear_layer_pipelined()
# mod()
# %%
# unroll
def linear_layer_unroll():
    concrete_type, L, D, O = float32, 1024, 1024, 1024
    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", W: "Ty[D, O]", B: "Ty[O]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        for i, j in dsl.grid(L, O):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D):
                Z[i, j] += X[i, k] * W[k, j]
    

    s_linear = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])
    s_linear.unroll("k")

    def top[Ty, L, D, O](a: "Ty[L, D]",W: "Ty[D, O]", B: "Ty[O]", b: "Ty[L, O]"):
        kernel_linear_layer[float32, 1024, 1024, 1024](a, W,B, b)

    s = allo.customize(top, instantiate=[concrete_type, L, D, O])
    s.compose(s_linear)

    return s.build(target="vitis_hls", mode="csyn", project="unroll.prj")

# mod = linear_layer_unroll()
# mod()

# %%
# partition
def linear_layer_partition():
    concrete_type, L, D, O = float32, 1024, 1024, 1024
    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", W: "Ty[D, O]", B: "Ty[O]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        for i, j in dsl.grid(L, O, name="outer_loop"):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D, name="inner_loop"):
                Z[i, j] += X[i, k] * W[k, j]
    

    s_linear = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])
    s_linear.partition(s_linear.W, partition_type=2, dim=1, factor=32)
    s_linear.partition(s_linear.X, partition_type=2, dim=2, factor=32)

    def top[Ty, L, D, O](a: "Ty[L, D]",W: "Ty[D, O]", B: "Ty[O]", b: "Ty[L, O]"):
        kernel_linear_layer[float32, 1024, 1024, 1024](a, W,B, b)

    s = allo.customize(top, instantiate=[concrete_type, L, D, O])
    s.compose(s_linear)

    return s.build(target="vitis_hls", mode="csyn", project="partition.prj")

# mod = linear_layer_partition()
# mod()
# %%
def linear_layer_optimized():
    concrete_type, L, D, O = float32,1024,1024,1024
    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", W: "Ty[D, O]", B: "Ty[O]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        buf_Z: Ty[O] = 0
        a: Ty = 0
        for i in dsl.grid(L, name="batch_loop"): 
            for j1 in dsl.grid(O, name="intialize_Z"):
                buf_Z[j1] =  B[j1]
            
            for k in dsl.grid(D, name="linear_compute"):
                a = X[i, k]
                for j2 in dsl.reduction(O, name="inner_prod"):
                    buf_Z[j2] += a * W[k, j2]

            for j3 in dsl.grid(O, name="get_value"):
                Z[i, j3] = buf_Z[j3]
    
    s_linear = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])
    s_linear.unroll("j1", factor=32)
    s_linear.pipeline("j1") 
    s_linear.unroll("j2", factor=32)
    s_linear.pipeline("j2") 
    s_linear.unroll("j3", factor=32)
    s_linear.pipeline("j3")  
    s_linear.partition(s_linear.buf_Z, partition_type=2, dim=1, factor=32)
    s_linear.partition(s_linear.Z, partition_type=2, dim=2, factor=32)
    # s_linear.partition(s_linear.W, partition_type=2, dim=2, factor=32)

    def top[Ty, L, D, O](a: "Ty[L, D]",  W: "Ty[D, O]", B: "Ty[O]", b: "Ty[L, O]"):
        kernel_linear_layer[float32, 1024, 1024, 1024](a, W, B, b)

    s = allo.customize(top, instantiate=[concrete_type, L, D, O])
    s.compose(s_linear)

    return s.build()

# mod= linear_layer_optimized()
# mod()

# %%    
def test_linear_layer():
    # Parameters for small test size
    L, D, O = 1024, 1024, 1024  # Match nn.Linear(2, 16)

    # Random input, weight, and bias initialization
    X = np.random.randn(L, D).astype(np.float32)
    W = np.random.randn(D, O).astype(np.float32)
    B = np.random.randn(O).astype(np.float32)
    allo_C = np.zeros((L, O), dtype=np.float32)
     
    # Instantiate the layer using systolic optimizations
    mod= linear_layer_optimized()
    mod(X, W,B,allo_C)
    # ref = numpy_linear_layer(X, W, B)
    linear_layer = nn.Linear(1024, 1024)
    with torch.no_grad():  # Avoid tracking gradients during this operation
        linear_layer.weight.copy_(torch.from_numpy(W.T))
        linear_layer.bias.copy_(torch.from_numpy(B.T))
    
    ref = linear_layer(torch.from_numpy(X)).detach().numpy()
    # Verify with NumPy reference
    # Verify with NumPy reference

    np.testing.assert_allclose(allo_C, ref, atol=1e-3)
    print("Test Passed!")


test_linear_layer()

# %%
