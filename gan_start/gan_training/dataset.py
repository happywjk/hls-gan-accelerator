   # %% 
import numpy as np
from torch.utils.data import Dataset
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import config
#    # %% 
# haha=os.listdir('/work/global/jw2777/gan_start/dataset/cartoon/train/minion/')
# hehe = "minions.png"
# hihi = os.path.join('/work/global/jw2777/gan_start/dataset/cartoon/train/minion/',hehe)
# image1 = Image.open(hihi)
# resize_image = image1.resize((512,512))
# resize_image = resize_image.convert("RGB") 
# image = np.array(resize_image)
# %%
class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size,config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        try:
            img = self.list_files[index]
            img_path = os.path.join(self.root_dir, img)
            image = Image.open(img_path).convert("RGB")
            tensor_image = self.transform(image)
            return tensor_image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((index + 1) % self.__len__()) 


        return tensor_image
