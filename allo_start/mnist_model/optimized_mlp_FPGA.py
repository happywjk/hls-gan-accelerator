# %%
import torch

import allo
import time
#print(torch.cuda.is_available())
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import os
from allo.ir.types import float32


import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from allo import dsl
from pytorch_optimization.linear import linear_layer
from pytorch_optimization.relu import relu

# %%                           1024 2 16  32  
def alloGenerator(concrete_type, L, D, O, H):
    def kernel_relu[Ty, L, D](X: "Ty[L, D]", Z:"Ty[L, D]"):
    # Apply ReLU element-wise
        for i, j in dsl.grid(L, D, name="relu_compute"):
            if X[i, j] > 0:
                Z[i, j] = X[i, j]
            else:
                Z[i, j] = 0.0
        

    s_relu = allo.customize(kernel_relu, instantiate=[concrete_type, L, O])
    s_relu.buffer_at(s_relu.Z, axis="i")
    s_relu.pipeline("j")
    def kernel_relu_2[Ty, L, D](X: "Ty[L, D]", Z:"Ty[L, D]"):
    # Apply ReLU element-wise
        for i, j in dsl.grid(L, D, name="relu_compute"):
            if X[i, j] > 0:
                Z[i, j] = X[i, j]
            else:
                Z[i, j] = 0.0
        

    s_relu_2 = allo.customize(kernel_relu_2, instantiate=[concrete_type, L, H])
    s_relu_2.buffer_at(s_relu_2.Z, axis="i")
    s_relu_2.pipeline("j")


    W_int_1 = np.random.rand(D, O).astype(np.float32)
    B_int_1 = np.random.rand(O).astype(np.float32)
    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        W: Ty[D, O] = W_int_1
        B: Ty[O]    = B_int_1
        for i, j in dsl.grid(L, O, name="linear_compute"):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D, name="inner_prod"):
                Z[i, j] += X[i, k] * W[k, j]
        
    s_linear = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])
    s_linear.buffer_at(s_linear.Z, axis="i")
    s_linear.pipeline("k")
    s_linear.partition(s_linear.Z, partition_type=1, dim=2, factor=2)

    W_int_2 = np.random.rand(O, H).astype(np.float32)
    B_int_2 = np.random.rand(H).astype(np.float32)
    def kernel_linear_layer_2[Ty, L, D, O](X: "Ty[L, D]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        W: Ty[D, O] = W_int_2
        B: Ty[O]    = B_int_2
        for i, j in dsl.grid(L, O, name="linear_compute"):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D, name="inner_prod"):
                Z[i, j] += X[i, k] * W[k, j]
        
    s_linear_2 = allo.customize(kernel_linear_layer_2, instantiate=[concrete_type, L, O, H])
    s_linear_2.buffer_at(s_linear_2.Z, axis="i")
    s_linear_2.pipeline("k")
    s_linear_2.partition(s_linear_2.Z, partition_type=1, dim=2, factor=2)

    W_int_3 = np.random.rand(H, D).astype(np.float32)
    B_int_3 = np.random.rand(D).astype(np.float32)
    def kernel_linear_layer_3[Ty, L, D, O](X: "Ty[L, D]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        W: Ty[D, O] = W_int_3
        B: Ty[O]    = B_int_3
        for i, j in dsl.grid(L, O, name="linear_compute"):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D, name="inner_prod"):
                Z[i, j] += X[i, k] * W[k, j]
        
    s_linear_3 = allo.customize(kernel_linear_layer_3, instantiate=[concrete_type, L, H, D])
    s_linear_3.buffer_at(s_linear_3.Z, axis="i")
    s_linear_3.pipeline("k")
    s_linear_3.partition(s_linear_3.Z, partition_type=1, dim=2, factor=2)

    def top[Ty, L, D, O, H](a: "Ty[L, D]",f: "Ty[L, D]"):
        # Forward pass for linear layer
        kernel_linear_layer(a,b)
        kernel_relu(b,c)
        kernel_linear_layer_2(c,d)
        kernel_relu_2(d,e)
        kernel_linear_layer_3(e,f)
        
    
    s = allo.customize(top, instantiate=[concrete_type, L, D, O, H])
    s.compse(s_linear)
    s.compose(s_relu)
    s.compse(s_linear_2)
    s.compose(s_relu_2)
    s.compse(s_linear_3)
    print(s.module)

alloGenerator(float32, 1024, 2, 16, 32)
# %%
#generator.load_state_dict(torch.load('/home/jw2777/generator_state_dict.pth'))
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x= self.fc1(x)
        x= F.relu(x)
        x= self.fc2(x)
        x= F.relu(x)
        x= self.fc3(x)
        return x
generator = SimpleGenerator()
# %%
example_inputs = [torch.randn(1024, 2)]

# %%

# %%
#total_time = 0.0  # Initialize total time
#for i in range(400):
total_time = 0.0  # Initialize total time

latent_space_samples = torch.randn(1024, 2)
b = np.zeros((1024, 16), dtype=np.float32)
c = np.zeros((1024, 16), dtype=np.float32)
d = np.zeros((1024, 32), dtype=np.float32)
f = np.zeros((1024, 32), dtype=np.float32)
g = np.zeros((1024, 2), dtype=np.float32)
golden = generator(latent_space_samples)
np_inputs = np.stack([x.detach().numpy() for x in latent_space_samples])
start_time = time.time()
allo_generator(np_inputs,b,c,d,f,g)
end_time = time.time()
total_time += (end_time - start_time)

print(f"Total time for initial 410 test cases: {total_time:.5f} seconds")


#torch.testing.assert_close(g, golden.detach().numpy())
#print("Success!")
# %%
