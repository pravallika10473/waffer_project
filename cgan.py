import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_EPOCHS = 200
LATENT_DIM = 100
NUM_CLASSES = 8
IMAGE_SIZE = 32
CHANNELS = 1

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Custom Dataset
class WaferDataset(Dataset):
    def __init__(self, wafer_maps, labels):
        self.wafer_maps = wafer_maps
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wafer_map = torch.tensor(self.wafer_maps[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return wafer_map, label

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, LATENT_DIM)
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM * 2, 256, 4, 1, 0, bias=False),
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

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        gen_input = gen_input.unsqueeze(2).unsqueeze(3)
        return self.model(gen_input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(NUM_CLASSES, IMAGE_SIZE * IMAGE_SIZE)
        
        self.model = nn.Sequential(
            nn.Conv2d(CHANNELS + 1, 64, 4, 2, 1, bias=False),
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

    def forward(self, img, labels):
        label_embedding = self.label_embedding(labels).view(labels.size(0), 1, IMAGE_SIZE, IMAGE_SIZE)
        d_input = torch.cat((img, label_embedding), 1)
        return self.model(d_input).view(-1, 1).squeeze(1)

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

# Loss function
criterion = nn.BCELoss()

# Load data
train_df = pd.read_pickle('/scratch/general/vast/u1475870/wafer_project/data/WM811K_training.pkl')
X_train = np.stack(train_df["waferMap"].values)
y_train = train_df["failureType"].values

train_dataset = WaferDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (real_images, labels) in enumerate(tqdm(train_loader)):
        batch_size = real_images.size(0)
        real_images, labels = real_images.to(device), labels.to(device)

        # Train Discriminator
        d_optimizer.zero_grad()

        # Real images
        real_validity = discriminator(real_images, labels)
        d_real_loss = criterion(real_validity, torch.ones_like(real_validity))

        # Fake images
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_images = generator(z, labels)
        fake_validity = discriminator(fake_images.detach(), labels)
        d_fake_loss = criterion(fake_validity, torch.zeros_like(fake_validity))

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_images = generator(z, labels)
        fake_validity = discriminator(fake_images, labels)

        # Generator loss
        g_loss = criterion(fake_validity, torch.ones_like(fake_validity))
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # Save generated images
    if (epoch + 1) % 10 == 0:
        generator.eval()
        with torch.no_grad():
            sample_z = torch.randn(NUM_CLASSES, LATENT_DIM).to(device)
            sample_labels = torch.arange(NUM_CLASSES).to(device)
            sample_images = generator(sample_z, sample_labels)
            
            fig, axs = plt.subplots(2, 4, figsize=(12, 6))
            for j in range(NUM_CLASSES):
                ax = axs[j // 4, j % 4]
                ax.imshow(sample_images[j].cpu().squeeze(), cmap='gray')
                ax.axis('off')
                ax.set_title(f'Class {j}')
            
            plt.tight_layout()
            plt.savefig(f'/scratch/general/vast/u1475870/wafer_project/outputs/generated_wafers_epoch_{epoch+1}.png')
            plt.close()
        generator.train()

# Save the models
torch.save(generator.state_dict(), '/scratch/general/vast/u1475870/wafer_project/outputs/generator.pth')
torch.save(discriminator.state_dict(), '/scratch/general/vast/u1475870/wafer_project/outputs/discriminator.pth')

print("Training completed and models saved.")