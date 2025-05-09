import torch



Device = "cuda" if torch.cuda.is_available() else "cpu"
print(Device)
Train_dir = "/work/global/jw2777/gan_start/dataset/cartoonset100k_jpg/0"
Test_dir = "/work/global/jw2777/gan_start/dataset/cartoonset100k_jpg/0"
Learning_rate = 1e-4
Batch_size = 32
Num_epochs = 100
image_size = 64
z_dim = 128
weight = 0.01
Load_model = False
Save_model = True

# CHECKPOINT_DISC = "disc.pth.tar"
# CHECKPOINT_GEN = "gen.pth.tar"