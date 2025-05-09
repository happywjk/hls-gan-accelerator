 
 
import matplotlib.pyplot as plt
import numpy as np
import torch
images = []
with open("output.data", "r") as f:
        for line in f:
            value = float(line.strip())
            images.append(value)
images = torch.tensor(images)
images = images.reshape(8, 3, 128, 128) 

images = (images + 1) / 2  # Normalize to [0, 1]

# Create a figure with 2 rows and 4 columns
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i, ax in enumerate(axes.flat):
    image = images[i].permute(1, 2, 0)  # Convert (C, H, W) -> (H, W, C)
    ax.imshow(image)
    ax.axis("off")  # Hide axes

plt.tight_layout()
plt.savefig(f"new/image_hardware1.png")