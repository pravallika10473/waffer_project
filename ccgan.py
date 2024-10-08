import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn.utils import spectral_norm
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0004
NUM_EPOCHS = 500
PRE_TRAIN_EPOCHS = 50
RANDOM_SEED = 42
NOISE_DIM = 100

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAMES = {
    0: "Center", 1: "Donut", 2: "Edge-Loc", 3: "Edge-Ring", 4: "Loc",
    5: "Random", 6: "Scratch", 7: "Near-full", 8: "none"
}

class AdaptiveLambda:
    def __init__(self, device, initial_lambda=0.1, alpha=0.001, min_lambda=0.01, max_lambda=10):
        self.device = device
        self.lambda_value = torch.tensor(initial_lambda, device=self.device)
        self.alpha = torch.tensor(alpha, device=self.device)
        self.min_lambda = torch.tensor(min_lambda, device=self.device)
        self.max_lambda = torch.tensor(max_lambda, device=self.device)

    def update(self, g_loss_adv, g_loss_class):
        ratio = g_loss_adv / (g_loss_class + 1e-8)  # Avoid division by zero
        self.lambda_value *= torch.exp(self.alpha * (ratio - 1))
        self.lambda_value = torch.clamp(self.lambda_value, self.min_lambda, self.max_lambda)
        return self.lambda_value.item()

class WaferDataset(Dataset):
    def __init__(self, wafer_maps, labels):
        self.wafer_maps = wafer_maps
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wafer_map = self.wafer_maps[idx]
        wafer_map = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return wafer_map, label

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.BatchNorm2d(128)),
            nn.Flatten(),
            spectral_norm(nn.Linear(128 * 8 * 8, 1024)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(1024, num_classes))
        )
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 8 * 8 * 128)),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 8, 8)),
            spectral_norm(nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.BatchNorm2d(64)),
            spectral_norm(nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.BatchNorm2d(128)),
            nn.Flatten(),
            spectral_norm(nn.Linear(128 * 8 * 8, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def prepare_data(X_train, y_train):
    train_dataset = WaferDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

def pretrain_classifier(classifier, train_loader, device):
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE_G)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(PRE_TRAIN_EPOCHS):
        total_loss = 0
        for wafers, labels in train_loader:
            wafers, labels = wafers.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(wafers)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Pre-training Epoch [{epoch+1}/{PRE_TRAIN_EPOCHS}], Avg Loss: {avg_loss:.4f}")

def generate_input(batch_size, noise_dim, num_classes, device):
    noise = torch.rand(batch_size, noise_dim, device=device) * 2 - 1
    class_indices = torch.randint(0, num_classes, (batch_size,), device=device)
    one_hot = nn.functional.one_hot(class_indices, num_classes=num_classes).float()
    input_tensor = torch.cat([noise, one_hot], dim=1)
    return input_tensor, class_indices

def create_mixed_batch(real_images, generator, noise_dim, num_classes, device):
    batch_size = real_images.size(0)
    num_real = torch.randint(1, batch_size, (1,)).item()
    num_fake = batch_size - num_real
    real_indices = torch.randperm(batch_size)[:num_real]
    real_batch = real_images[real_indices]
    input_tensor, _ = generate_input(num_fake, noise_dim, num_classes, device)
    fake_batch = generator(input_tensor)
    mixed_batch = torch.cat([real_batch, fake_batch], dim=0)
    mixed_labels = torch.cat([torch.ones(num_real, 1), torch.zeros(num_fake, 1)], dim=0).to(device)
    shuffle_indices = torch.randperm(batch_size)
    mixed_batch = mixed_batch[shuffle_indices]
    mixed_labels = mixed_labels[shuffle_indices]
    return mixed_batch, mixed_labels

def train_models(classifier, generator, discriminator, train_loader, device):
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(0.5, 0.999))
    
    generator_scheduler = ExponentialLR(generator_optimizer, gamma=0.99)
    discriminator_scheduler = ExponentialLR(discriminator_optimizer, gamma=0.99)
    
    criterion = nn.BCELoss()
    classifier_criterion = nn.CrossEntropyLoss()

    adaptive_lambda = AdaptiveLambda(device)

    pretrain_classifier(classifier, train_loader, device)
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        d_losses, g_losses = [], []

        for batch_idx, (wafers, _) in enumerate(train_loader):
            batch_size = wafers.size(0)
            wafers = wafers.to(device)
            
            # Train Discriminator
            discriminator_optimizer.zero_grad()
            mixed_batch, mixed_labels = create_mixed_batch(wafers, generator, NOISE_DIM, len(CLASS_NAMES), device)
            mixed_output = discriminator(mixed_batch)
            d_loss = criterion(mixed_output, mixed_labels)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            discriminator_optimizer.step()
            
            # Train Generator
            generator_optimizer.zero_grad()
            fake_input, class_indices = generate_input(batch_size, NOISE_DIM, len(CLASS_NAMES), device)
            fake_images = generator(fake_input)
            fake_output = discriminator(fake_images)
            g_loss_adv = criterion(fake_output, torch.ones(batch_size, 1).to(device))
            
            fake_classifications = classifier(fake_images)
            g_loss_class = classifier_criterion(fake_classifications, class_indices)
            
            lambda_value = adaptive_lambda.update(g_loss_adv, g_loss_class)
            g_loss = g_loss_adv + lambda_value * g_loss_class
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            generator_optimizer.step()
            
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}/{len(train_loader)}], "
                      f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
                      f"Adv Loss: {g_loss_adv.item():.4f}, Class Loss: {g_loss_class.item():.4f}, "
                      f"Lambda: {lambda_value:.4f}")
        
        generator_scheduler.step()
        discriminator_scheduler.step()
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Duration: {epoch_duration:.2f}s")
        print(f"D loss: {np.mean(d_losses):.4f}, G loss: {np.mean(g_losses):.4f}")
        
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, device, output_dir)

def generate_and_save_images(generator, epoch, device, output_dir):
    generator.eval()
    with torch.no_grad():
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f"Generated Wafer Maps - Epoch {epoch}", fontsize=16)

        for class_idx in range(8):
            input_tensor = torch.zeros(1, NOISE_DIM + len(CLASS_NAMES), device=device)
            input_tensor[0, :NOISE_DIM] = torch.rand(1, NOISE_DIM, device=device) * 2 - 1
            input_tensor[0, NOISE_DIM + class_idx] = 1
            
            fake_image = generator(input_tensor)
            img_array = fake_image[0, 0].cpu().numpy()
            
            row = class_idx // 4
            col = class_idx % 4
            ax = axes[row, col]
            im = ax.imshow(img_array, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Class: {CLASS_NAMES[class_idx]}")
            
            plt.colorbar(im, ax=ax, orientation='vertical', pad=0.05, fraction=0.046)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(output_dir, 'ccgan_images', f'epoch_{epoch:04d}_wafer_maps.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    
    generator.train()
    print(f"Images for epoch {epoch} saved to {os.path.join(output_dir, 'ccgan_images')}")

def generate_and_save_final_images(generator, classifier, device, output_dir, num_images_per_class=10):
    generator.eval()
    classifier.eval()
    
    with torch.no_grad():
        for class_idx in range(len(CLASS_NAMES)):
            input_tensor = torch.zeros(num_images_per_class, NOISE_DIM + len(CLASS_NAMES), device=device)
            input_tensor[:, :NOISE_DIM] = torch.rand(num_images_per_class, NOISE_DIM, device=device) * 2 - 1
            input_tensor[:, NOISE_DIM + class_idx] = 1
            
            fake_images = generator(input_tensor)
            classifications = classifier(fake_images)
            _, predicted_classes = torch.max(classifications, 1)
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            fig.suptitle(f"Generated images for class: {CLASS_NAMES[class_idx]}")
            
            for i, ax in enumerate(axes.flat):
                ax.imshow(fake_images[i, 0].cpu().numpy(), cmap='gray')
                ax.axis('off')
                ax.set_title(f"Classified as: {CLASS_NAMES[predicted_classes[i].item()]}")
            
            output_path = os.path.join(output_dir, 'final_images', f'class_{class_idx}_{CLASS_NAMES[class_idx]}.png')
            plt.savefig(output_path)
            plt.close()
    
    generator.train()
    classifier.train()
    print(f"Final images saved to {os.path.join(output_dir, 'final_images')}")

# Define the output directory
output_dir = '/scratch/general/vast/u1475870/wafer_project/output/'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'ccgan_images'), exist_ok=True)

# Load data
train_df = pd.read_pickle('/scratch/general/vast/u1475870/wafer_project/data/WM811K_training.pkl')

X_train = np.stack(train_df["waferMap"].values).astype('float32')
X_train = X_train / np.max(X_train)
y_train = train_df["failureType"].values.astype('int64')

train_loader = prepare_data(X_train, y_train)

num_classes = len(CLASS_NAMES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classifier = Classifier(num_classes).to(device)
generator = Generator(NOISE_DIM + num_classes).to(device)
discriminator = Discriminator().to(device)

# Training loop
start_time = time.time()
train_models(classifier, generator, discriminator, train_loader, device)
end_time = time.time()
total_duration = end_time - start_time
print(f"Total training time: {total_duration:.2f} seconds")

# Save the models
torch.save(classifier.state_dict(), os.path.join(output_dir, 'ccgan_classifier.pth'))
torch.save(generator.state_dict(), os.path.join(output_dir, 'ccgan_generator.pth'))
torch.save(discriminator.state_dict(), os.path.join(output_dir, 'ccgan_discriminator.pth'))
print(f"Models saved to {output_dir}")

# Generate and save final images
os.makedirs(os.path.join(output_dir, 'final_images'), exist_ok=True)
generate_and_save_final_images(generator, classifier, device, output_dir)