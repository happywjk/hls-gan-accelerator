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


# %%                           1024 2 16  32  
def alloGenerator():
    concrete_type = float32
    L = 1024
    D = 2
    O = 16
    H = 32
    def kernel_relu[Ty, L, D](X: "Ty[L, D]", Z:"Ty[L, D]"):
    # Apply ReLU element-wise
        for i, j in dsl.grid(L, D, name="relu_compute"):
            if X[i, j] > 0:
                Z[i, j] = X[i, j]
            else:
                Z[i, j] = 0.0
        

    s_relu = allo.customize(kernel_relu, instantiate=[concrete_type, L, O])
    s_relu2 = allo.customize(kernel_relu, instantiate=[concrete_type, L, H])
    s_relu.buffer_at(s_relu.Z, axis="i")
    s_relu.pipeline("j")


    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", W: "Ty[D, O]",B: "Ty[O]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        for i, j in dsl.grid(L, O, name="linear_compute"):
            Z[i, j] = B[j]  # Initialize with bias
            for k in dsl.reduction(D, name="inner_prod"):
                Z[i, j] += X[i, k] * W[k, j]
        
    s_linear_1 = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])
    s_linear_1.buffer_at(s_linear_1.Z, axis="i")
    s_linear_1.pipeline("k")
    s_linear_1.partition(s_linear_1.Z, partition_type=1, dim=2, factor=2)

    s_linear_2 = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, O, H])
    s_linear_2.buffer_at(s_linear_2.Z, axis="i")
    s_linear_2.pipeline("k")
    s_linear_2.partition(s_linear_2.Z, partition_type=1, dim=2, factor=2)

    s_linear_3 = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, H, D])
    s_linear_3.buffer_at(s_linear_3.Z, axis="i")
    s_linear_3.pipeline("k")
    s_linear_3.partition(s_linear_3.Z, partition_type=1, dim=2, factor=2)

    W_int_1 = np.random.rand(D, O).astype(np.float32)
    B_int_1 = np.random.rand(O).astype(np.float32)
    W_int_2 = np.random.rand(O, H).astype(np.float32)
    B_int_2 = np.random.rand(H).astype(np.float32)
    W_int_3 = np.random.rand(H, D).astype(np.float32)
    B_int_3 = np.random.rand(D).astype(np.float32)
    def top[Ty, L, D,O,H](a: "Ty[L, D]",f: "Ty[L, D]"):

        W_1: Ty[D, O] = W_int_1
        B_1:   Ty[O]    = B_int_1
        W_2: Ty[O, H] = W_int_2
        B_2:   Ty[H]    = B_int_2
        W_3: Ty[H, D] = W_int_3
        B_3:   Ty[D]    = B_int_3
        # Forward pass for linear layer
        b:float32[1024,16] = 0
        c:float32[1024,16] = 0
        d:float32[1024,32] = 0
        e:float32[1024,32] = 0
        kernel_linear_layer[float32, 1024,2,16,"linear_1"](a,W_1,B_1,b)
        kernel_relu[float32,1024,16,"relu_1"](b,c)
        kernel_linear_layer[float32, 1024,16,32,"linear_2"](c,W_2,B_2,d)
        kernel_relu[float32,1024,32,"relu_2"](d,e)
        kernel_linear_layer[float32, 1024,32,2,"linear_3"](e,W_3,B_3,f)


    s = allo.customize(top, instantiate=[concrete_type, L, D,O,H])
    s.compose(s_linear_1, id="linear_1")
    s.compose(s_relu, id="relu_1")
    s.compose(s_linear_2, id="linear_2")
    s.compose(s_relu2, id="relu_2")
    s.compose(s_linear_3, id="linear_3")
    print(s.module)

alloGenerator()
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
