# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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
from torchvision import datasets
import torchvision.transforms as transforms
from json import JSONEncoder
import json
# %%

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 20

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# choose the training and test datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                                  download=True, transform=transform)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
    num_workers=num_workers)
# %%
# dataiter = iter(test_loader)
# images, labels = next(dataiter)
# images = images.view(-1, 28*28)
# for data, target in test_loader:
#     data
#     target
#     break
state_dict = torch.load("/work/global/jw2777/allo_start/model_state.pth")

# Define the layer keys
keys = [
    "fc1.weight", "fc1.bias",
    "fc2.weight", "fc2.bias",
    "fc3.weight", "fc3.bias"
]

# Initialize an empty dictionary to store the NumPy arrays
layer_data = {}

# %%
# Iterate through the keys and extract tensors, converting them to NumPy arrays
for key in keys:
    if key in state_dict:
        layer_data[key] = state_dict[key].numpy()
    else:
        raise KeyError(f"Key '{key}' not found in the state dictionary!")

# Assign the weights and biases to specific variables
layer1_weights = np.ascontiguousarray(layer_data["fc1.weight"].T).astype(np.float32)
layer1_bias = np.ascontiguousarray(layer_data["fc1.bias"]).astype(np.float32)
layer2_weights = np.ascontiguousarray(layer_data["fc2.weight"].T).astype(np.float32)
layer2_bias = np.ascontiguousarray(layer_data["fc2.bias"]).astype(np.float32)
layer3_weights = np.ascontiguousarray(layer_data["fc3.weight"].T).astype(np.float32)
layer3_bias = np.ascontiguousarray(layer_data["fc3.bias"]).astype(np.float32)

# Print confirmation and optionally the shapes
print("All layer weights and biases loaded successfully!")
print(f"fc1 weights shape: {layer1_weights.shape}, bias shape: {layer1_bias.shape}")
print(f"fc2 weights shape: {layer2_weights.shape}, bias shape: {layer2_bias.shape}")
print(f"fc3 weights shape: {layer3_weights.shape}, bias shape: {layer3_bias.shape}")
# %%
# def alloGenerator():
#     concrete_type = float32
#     L = 20
#     D = 28*28
#     O = 512
#     H = 10
#     def kernel_relu[Ty, L, D](X: "Ty[L, D]", Z:"Ty[L, D]"):
#     # Apply ReLU element-wise
#         for i, j in dsl.grid(L, D, name="relu_compute"):
#             if X[i, j] > 0:
#                 Z[i, j] = X[i, j]
#             else:
#                 Z[i, j] = 0.0
        

#     s_relu = allo.customize(kernel_relu, instantiate=[concrete_type, L, O])
#     s_relu2 = allo.customize(kernel_relu, instantiate=[concrete_type, L, O])
#     # s_relu.buffer_at(s_relu.Z, axis="i")
#     # s_relu.pipeline("j")


#     def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", W: "Ty[D, O]",B: "Ty[O]", Z:"Ty[L, O]"):
#         # Forward pass for linear layer
#         for i, j in dsl.grid(L, O, name="linear_compute"):
#             Z[i, j] = B[j]  # Initialize with bias
#             for k in dsl.reduction(D, name="inner_prod"):
#                 Z[i, j] += X[i, k] * W[k, j]
        
#     s_linear_1 = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, D, O])
#     # s_linear_1.buffer_at(s_linear_1.Z, axis="i")
#     # s_linear_1.pipeline("k")
#     # s_linear_1.partition(s_linear_1.Z, partition_type=1, dim=2, factor=2)

#     s_linear_2 = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, O, O])
#     # s_linear_2.buffer_at(s_linear_2.Z, axis="i")
#     # s_linear_2.pipeline("k")
#     # s_linear_2.partition(s_linear_2.X, partition_type=1, dim=2, factor=2)
#     # s_linear_2.partition(s_linear_2.Z, partition_type=1, dim=2, factor=2)

#     s_linear_3 = allo.customize(kernel_linear_layer, instantiate=[concrete_type, L, O, H])
#     # s_linear_3.buffer_at(s_linear_3.Z, axis="i")
#     # s_linear_3.pipeline("k")
#     # s_linear_3.partition(s_linear_3.X, partition_type=1, dim=2, factor=2)


#     def top[Ty, L, D, O, H](
#         a: "Ty[L, D]",
#         W_1: "Ty[D, O]",
#         B_1: "Ty[O]",
#         W_2: "Ty[O, O]",
#         B_2: "Ty[O]",
#         W_3: "Ty[O, H]",
#         B_3: "Ty[H]",
#         f: "Ty[L, H]"
#     ):
#         # Forward pass for linear layer
#         b:float32[20,512] = 0
#         c:float32[20,512] = 0
#         d:float32[20,512] = 0
#         e:float32[20,512] = 0
#         kernel_linear_layer[float32, 20,28*28,512,"linear_1"](a,W_1,B_1,b)
#         kernel_relu[float32,20,512,"relu_1"](b,c)
#         kernel_linear_layer[float32, 20,512,512,"linear_2"](c,W_2,B_2,d)
#         kernel_relu[float32,20,512,"relu_2"](d,e)
#         kernel_linear_layer[float32, 20,512,10,"linear_3"](e,W_3,B_3,f)


#     s = allo.customize(top, instantiate=[concrete_type, L, D,O,H])
#     s.compose(s_linear_1, id="linear_1")
#     s.compose(s_relu, id="relu_1")
#     s.compose(s_linear_2, id="linear_2")
#     s.compose(s_relu2, id="relu_2")
#     s.compose(s_linear_3, id="linear_3")
#     # print(s.module)
#     # return s.build(target="llvm")
#     # return s.build(target="vitis_hls", mode="csyn", project="mnist_mlp_unoptimized.prj")

# %%
def alloGenerator():
    concrete_type = float32
    L = 20
    D = 28*28
    O = 512
    H = 10
    def kernel_relu[Ty, L, D](X: "Ty[L, D]", Z:"Ty[L, D]"):
    # Apply ReLU element-wise
        for i, j in dsl.grid(L, D, name="relu_compute"):
            if X[i, j] > 0:
                Z[i, j] = X[i, j]
            else:
                Z[i, j] = 0.0
        

    s_relu = allo.customize(kernel_relu, instantiate=[concrete_type, L, O])
    s_relu2 = allo.customize(kernel_relu, instantiate=[concrete_type, L, O])
    s_relu.buffer_at(s_relu.Z, axis="i")
    s_relu.pipeline("j")


    def kernel_linear_layer[Ty, L, D, O](X: "Ty[L, D]", Z:"Ty[L, O]"):
        # Forward pass for linear layer
        W_1: Ty[D, O] = layer1_weights
        B_1: Ty[O] = layer1_bias
        buf_Z: Ty[O] = 0
        a: Ty = 0
        for i in dsl.grid(L, name="batch_loop"): 
            for j1 in dsl.grid(O, name="intialize_Z"):
                buf_Z[j1] = B_1[j1]
            
            for k in dsl.grid(D, name="linear_compute"):
                a = X[i, k]
                for j2 in dsl.reduction(O, name="inner_prod"):
                    buf_Z[j2] += a * W_1[k, j2]

            for j3 in dsl.grid(O, name="get_value"):
                Z[i, j3] = buf_Z[j3]
    
    def kernel_linear_layer_2[Ty, L, D, O](X: "Ty[L, D]", Z:"Ty[L, O]"):
        W_2: Ty[D, O] = layer2_weights
        B_2: Ty[O] = layer2_bias
        # Forward pass for linear layer
        buf_Z: Ty[O] = 0
        a: Ty = 0
        for i in dsl.grid(L, name="batch_loop"): 
            for j1 in dsl.grid(O, name="intialize_Z"):
                buf_Z[j1] = B_2[j1]
            
            for k in dsl.grid(D, name="linear_compute"):
                a = X[i, k]
                for j2 in dsl.reduction(O, name="inner_prod"):
                    buf_Z[j2] += a * W_2[k, j2]

            for j3 in dsl.grid(O, name="get_value"):
                Z[i, j3] = buf_Z[j3]

    def kernel_linear_layer_3[Ty, L, D, O](X: "Ty[L, D]", Z:"Ty[L, O]"):
        W_3: Ty[D, O] = layer3_weights
        B_3: Ty[O] = layer3_bias
        # Forward pass for linear layer
        buf_Z: Ty[O] = 0
        a: Ty = 0
        for i in dsl.grid(L, name="batch_loop"): 
            for j1 in dsl.grid(O, name="intialize_Z"):
                buf_Z[j1] = B_3[j1]
            
            for k in dsl.grid(D, name="linear_compute"):
                a = X[i, k]
                for j2 in dsl.reduction(O, name="inner_prod"):
                    buf_Z[j2] += a * W_3[k, j2]

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
    # s_linear.partition(s_linear.W_1, partition_type=2, dim=2, factor=32)
        

    s_linear_2 = allo.customize(kernel_linear_layer_2, instantiate=[concrete_type, L, O, O])
    s_linear_2.unroll("j1", factor=32)
    s_linear_2.pipeline("j1") 
    s_linear_2.unroll("j2", factor=32)
    s_linear_2.pipeline("j2") 
    s_linear_2.unroll("j3", factor=32)
    s_linear_2.pipeline("j3")  
    s_linear_2.partition(s_linear_2.buf_Z, partition_type=2, dim=1, factor=32)
    s_linear_2.partition(s_linear_2.Z, partition_type=2, dim=2, factor=32)
    # s_linear_2.partition(s_linear_2.W_2, partition_type=2, dim=2, factor=32)

    s_linear_3 = allo.customize(kernel_linear_layer_3, instantiate=[concrete_type, L, O, H])
    s_linear_3.unroll("j1", factor=32)
    s_linear_3.pipeline("j1") 
    s_linear_3.unroll("j2", factor=32)
    s_linear_3.pipeline("j2") 
    s_linear_3.unroll("j3", factor=32)
    s_linear_3.pipeline("j3")  
    s_linear_3.partition(s_linear_3.buf_Z, partition_type=2, dim=1, factor=32)
    s_linear_3.partition(s_linear_3.Z, partition_type=2, dim=2, factor=32)
    # s_linear_3.partition(s_linear_3.W_3, partition_type=2, dim=2, factor=32)


    def top[Ty, L, D, O, H](
        a: "Ty[L, D]",
        f: "Ty[L, H]"
    ):
        # Forward pass for linear layer
        b:float32[20,512] = 0
        c:float32[20,512] = 0
        d:float32[20,512] = 0
        e:float32[20,512] = 0
        kernel_linear_layer[float32, 20,28*28,512,"linear_1"](a,b)
        kernel_relu[float32,20,512,"relu_1"](b,c)
        kernel_linear_layer_2[float32, 20,512,512,"linear_2"](c,d)
        kernel_relu[float32,20,512,"relu_2"](d,e)
        kernel_linear_layer_3[float32, 20,512,10,"linear_3"](e,f)


    s = allo.customize(top, instantiate=[concrete_type, L, D,O,H])
    s.compose(s_linear, id="linear_1")
    s.compose(s_relu, id="relu_1")
    s.compose(s_linear_2, id="linear_2")
    s.compose(s_relu2, id="relu_2")
    s.compose(s_linear_3, id="linear_3")
    print(s.module)
    return s.build(target="vitis_hls", mode="hw", project="mnist_mlp_true_partition.prj")
mod = alloGenerator()
# %%
# %%

## Define the NN architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(512, 512)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(512, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # flatten image input
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))

                # Dropout after the first layer
        x = self.dropout(x)
        
        # Second layer with ReLU
        x = F.relu(self.fc2(x))
        
        # Dropout after the second layer
        x = self.dropout(x)
        
        # Third layer (output logits)
        x = self.fc3(x)
        
        return x

# initialize the NN
model = MLP()

criterion = nn.CrossEntropyLoss()
model.load_state_dict(torch.load("/work/global/jw2777/allo_start/model_state.pth"))
# %%
## Define the NN architecture

# %%
model.eval()
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
total_time = 0.0
for data, target in test_loader:
    # forward pass: compute predicted outputs by passing inputs to the model
    data= data.view(-1, 28*28)
    allo_data = np.ascontiguousarray(data.detach().numpy()).astype(np.float32)
    output = np.zeros((20, 10)).astype(np.float32)

    start_time = time.time()
    mod(allo_data, output)
    end_time = time.time()
    total_time += (end_time - start_time)
    output_pytorch = model(data)
    np.testing.assert_allclose(output, output_pytorch.detach().numpy(), rtol=1e-5, atol=1e-5)
    output = torch.tensor(output)
    # calculate the loss
    loss = criterion(output, target)
    # update test loss 
    test_loss += loss.item()*20
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    # calculate test accuracy for each object class
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# calculate and print avg test loss
test_loss = test_loss/len(test_loader.dataset)
print(f"Total time for mnist test dataset: {total_time:.5f} seconds")
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
# %%
