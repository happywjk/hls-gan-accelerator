# %%
import torch

import allo
import time
print(torch.cuda.is_available())
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import math
import matplotlib.pyplot as plt
import numpy as np
from random import uniform

# %%
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
generator.load_state_dict(torch.load('/home/jw2777/generator_state_dict.pth'))

# %%
example_inputs = [torch.randn(5000, 2)]
output_data = np.zeros((5000, 2))
llvm_mod = allo.frontend.from_pytorch(
    generator, example_inputs=example_inputs, 
    verbose=False, target="vitis_hls", mode="hw", project="gan_hw.prj")
total_time = 0.0  # Initialize total time

latent_space_samples = torch.randn(5000, 2)
golden = generator(latent_space_samples)
np_inputs = np.stack([x.detach().numpy() for x in latent_space_samples])
start_time = time.time()
llvm_mod(np_inputs, output_data)
end_time = time.time()
total_time += (end_time - start_time)
# %%

'''
llvm_mod = allo.frontend.from_pytorch(
    generator, example_inputs=example_inputs, 
    verbose=False, target="vitis_hls", mode="hw", project="gan_hw.prj")
llvm_mod(np_inputs, output_data)

print("FPGA Output Shape:", res)
print("Golden Output Shape:", golden.detach().numpy().shape)
'''
golden = golden.to(torch.float64)
torch.testing.assert_close(output_data, golden.detach().numpy())
print("Success!")
# %%

