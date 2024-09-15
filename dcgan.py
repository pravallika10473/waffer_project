import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 200
LATENT_DIM = 100
IMAGE_SIZE = 32
CHANNELS = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WaferDataset(Dataset):
    def __init__(self, wafer_maps):
        self.wafer_maps = wafer_maps
    
    def __len__(self):
        return len(self.wafer_maps)
    
    def __getitem__(self, idx):
        wafer_map = torch.tensor(self.wafer_maps[idx], dtype=torch.float32).unsqueeze(0)
        return wafer_map

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_dcgan(dataloader, num_epochs):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, real_images in enumerate(tqdm(dataloader)):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            # Train Discriminator
            d_optimizer.zero_grad()
            
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output = discriminator(real_images)
            d_loss_real = criterion(output, label)
            d_loss_real.backward()
            
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = generator(noise)
            label.fill_(0)
            output = discriminator(fake_images.detach())
            d_loss_fake = criterion(output, label)
            d_loss_fake.backward()
            
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            label.fill_(1)
            output = discriminator(fake_images)
            g_loss = criterion(output, label)
            g_loss.backward()
            g_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

        # Save generated images
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Generated Images at Epoch {epoch+1}")
            plt.imshow(np.transpose(make_grid(fake_images, padding=2, normalize=True), (1, 2, 0)))
            plt.savefig(output_dir + 'dcgan_images' + f'dcgan_wafers_epoch_{epoch+1}.png')
            plt.close()

    return generator, discriminator

def make_grid(tensor, padding=2, normalize=True):
    """Make a grid of images."""
    nmaps = tensor.size(0)
    xmaps = min(8, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new_full((3, height * ymaps + padding, width * xmaps + padding), 255)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    if normalize:
        grid = grid.float().div(255)
    return grid

if __name__ == "__main__":
    # Define the output directory
    output_dir = '/scratch/general/vast/u1475870/wafer_project/outputs/'

    # Load data
    train_df = pd.read_pickle('/scratch/general/vast/u1475870/wafer_project/data/WM811K_training.pkl')
    X_train = np.stack(train_df["waferMap"].values)

    train_dataset = WaferDataset(X_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Train DCGAN
    generator, discriminator = train_dcgan(train_loader, NUM_EPOCHS)

    # Save models
    torch.save(generator.state_dict(), output_dir + 'dcgan_generator.pth')
    torch.save(discriminator.state_dict(), output_dir + 'dcgan_discriminator.pth')

    print("Training completed and models saved.")