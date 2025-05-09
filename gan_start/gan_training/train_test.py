#%% packages
import torch
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import MapDataset
import config
from discriminator import Discriminator
from generator import Generator
from generator import initialize_weights
import numpy as np

# Hyperparameters for WGAN-GP
LAMBDA_GP = 10  # Gradient penalty coefficient
CRITIC_ITERATIONS = 5  # Number of critic updates per generator update

#%% Gradient Penalty Function
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Compute gradients
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

#%% Training Function for WGAN-GP
def train_fn(epoch, critic, gen, dataset, critic_opt, gen_opt, noise):
    data_loader = DataLoader(dataset, batch_size=config.Batch_size, shuffle=True)
    loop = tqdm.tqdm(data_loader)
    d_total_loss = 0
    g_total_loss = 0

    for index, img in enumerate(loop):
        img = img.to(config.Device)
        
        cur_batch_size = img.shape[0]
        
        # Train Critic multiple times per generator update
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, config.z_dim, 1, 1).to(config.Device)
            fake = gen(noise)

            critic_real = critic(img).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, img, fake, device=config.Device)

            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp

            critic_opt.zero_grad()
            loss_critic.backward(retain_graph=True)
            critic_opt.step()

        d_total_loss += loss_critic.item()

        # Train Generator
        gen_opt.zero_grad()
        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        loss_gen.backward()
        gen_opt.step()
        
        g_total_loss += loss_gen.item()

        # Print losses occasionally and print to tensorboard
        if index % 100 == 0 and index > 0:
            print(
                f"Epoch [{epoch}/{config.Num_epochs}] Batch {index}/{len(data_loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

# %% Main function
def main():
    critic = Discriminator().to(config.Device)  # Critic (Discriminator in WGAN)
    gen = Generator().to(config.Device)
    initialize_weights(gen)
    initialize_weights(critic)

    critic_opt = torch.optim.Adam(critic.parameters(), lr=config.Learning_rate, betas=(0.0, 0.9))
    gen_opt = torch.optim.Adam(gen.parameters(), lr=config.Learning_rate, betas=(0.0, 0.9))

    val_noise = torch.randn((8, config.z_dim, 1, 1)).to(config.Device)
    dataset = MapDataset(config.Train_dir)

    if config.Load_model:
        print("Loading model")
        gen.load_state_dict(torch.load('model1/gen.pth'))
        critic.load_state_dict(torch.load('model1/disc.pth'))
        gen_opt.load_state_dict(torch.load('model1/gen_opt.pth'))
        critic_opt.load_state_dict(torch.load('model1/disc_opt.pth'))

    for epoch in range(config.Num_epochs):
        print(f"Epoch: {epoch}")
        noise = torch.randn((config.Batch_size, config.z_dim, 1, 1)).to(config.Device)
        train_fn(epoch, critic, gen, dataset, critic_opt, gen_opt, noise)

        if config.Save_model:
            print("Saving model")
            torch.save(gen.state_dict(), 'model1/gen.pth')
            torch.save(critic.state_dict(), 'model1/disc.pth')
            torch.save(gen_opt.state_dict(), 'model1/gen_opt.pth')
            torch.save(critic_opt.state_dict(), 'model1/disc_opt.pth')
        
        if epoch % 1 == 0:
            with torch.no_grad():
                images = gen(val_noise).detach().cpu()  # Shape: (8, C, H, W)
                images = (images + 1) / 2  # Normalize to [0, 1]

                # Create a figure with 2 rows and 4 columns
                fig, axes = plt.subplots(2, 4, figsize=(12, 6))

                for i, ax in enumerate(axes.flat):
                    image = images[i].permute(1, 2, 0)  # Convert (C, H, W) -> (H, W, C)
                    ax.imshow(image)
                    ax.axis("off")  # Hide axes

                plt.tight_layout()
                plt.savefig(f"new/image_{epoch}.png")

if __name__ == "__main__":
    main()
