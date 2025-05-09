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

from pytorch_optimization.linear import linear_layer
from pytorch_optimization.relu import relu

# %%
class llvmGenerator():
    def __init__(self):
        self.fc1 = linear_layer(float32, 1024, 2, 16)
        self.re1 = relu(float32, 1024, 16)
        self.fc2 = linear_layer(float32, 1024, 16, 32)
        self.re2 = relu(float32, 1024, 32)
        self.fc3 = linear_layer(float32, 1024, 32, 2)

    def forward(self, a,b,c,d,f,g):
        self.fc1(a,b)
        self.re1(b,c)
        self.fc2(c,d)
        self.re2(d,f)
        self.fc3(f,g)
    
    def __call__(self,a,b,c,d,f,g):
        return self.forward(a,b,c,d,f,g)

allo_generator = llvmGenerator()
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
for i in range(410):
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
