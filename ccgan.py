import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
BATCH_SIZE = 128
LEARNING_RATE = 0.0002
NUM_EPOCHS = 1000
PRE_TRAIN_EPOCHS = 100
RANDOM_SEED = 42
NOISE_DIM = 100
LAMBDA = 1  # Weight for classifier loss in generator

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAMES = {
    0: "Center", 1: "Donut", 2: "Edge-Loc", 3: "Edge-Ring", 4: "Loc",
    5: "Random", 6: "Scratch", 7: "Near-full", 8: "none"
}

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
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8 * 8 * 128),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def prepare_data(X_train, y_train):
    train_dataset = WaferDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader

def pretrain_classifier(classifier, train_loader, device):
    optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
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
    # Generate noise in range [-1, 1]
    noise = torch.rand(batch_size, noise_dim, device=device) * 2 - 1
    # Generate random class indices
    class_indices = torch.randint(0, num_classes, (batch_size,), device=device)
    # Create one-hot encoded vectors
    one_hot = nn.functional.one_hot(class_indices, num_classes=num_classes).float()
    # Concatenate noise and one-hot vectors
    input_tensor = torch.cat([noise, one_hot], dim=1)
    return input_tensor, class_indices

def train_models(classifier, generator, discriminator, train_loader, device):
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()
    classifier_criterion = nn.CrossEntropyLoss()

    # Pre-train the classifier
    pretrain_classifier(classifier, train_loader, device)
    
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        d_losses, g_losses = [], []

        for wafers, _ in train_loader:
            batch_size = wafers.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            wafers = wafers.to(device)
            
            # Train Discriminator
            discriminator_optimizer.zero_grad()
            real_output = discriminator(wafers)
            d_loss_real = criterion(real_output, real_labels)
            
            input_tensor, class_indices = generate_input(batch_size, NOISE_DIM, len(CLASS_NAMES), device)
            fake_images = generator(input_tensor)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            discriminator_optimizer.step()
            
            # Train Generator
            generator_optimizer.zero_grad()
            fake_output = discriminator(fake_images)
            g_loss_adv = criterion(fake_output, real_labels)
            
            fake_classifications = classifier(fake_images)
            g_loss_class = classifier_criterion(fake_classifications, class_indices)
            
            g_loss = g_loss_adv + LAMBDA * g_loss_class
            g_loss.backward()
            generator_optimizer.step()
            
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Duration: {epoch_duration:.2f}s")
        print(f"D loss: {np.mean(d_losses):.4f}, G loss: {np.mean(g_losses):.4f}")
        
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, device, output_dir)

def generate_and_save_images(generator, epoch, device, output_dir):
    generator.eval()
    with torch.no_grad():
        input_tensor = torch.zeros(16, NOISE_DIM + len(CLASS_NAMES), device=device)
        input_tensor[:, :NOISE_DIM] = torch.rand(16, NOISE_DIM, device=device) * 2 - 1
        for i in range(8):
            input_tensor[i*2:(i+1)*2, NOISE_DIM + i] = 1
        fake_images = generator(input_tensor)
    
    fig = plt.figure(figsize=(4, 4))
    for i in range(fake_images.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(fake_images[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    
    output_path = os.path.join(output_dir, 'ccgan_images', f'ccgan_images_epoch_{epoch:04d}.png')
    plt.savefig(output_path)
    plt.close()
    generator.train()

def generate_and_save_final_images(generator, classifier, device, output_dir, num_images_per_class=10):
    generator.eval()
    classifier.eval()
    
    with torch.no_grad():
        for class_idx in range(len(CLASS_NAMES)):
            input_tensor = torch.zeros(num_images_per_class, NOISE_DIM + len(CLASS_NAMES), device=device)
            input_tensor[:, :NOISE_DIM] = torch.rand(num_images_per_class, NOISE_DIM, device=device) * 2 - 1
            input_tensor[:, NOISE_DIM + class_idx] = 1
            
            fake_images = generator(input_tensor)
            
            # Classify the generated images
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
os.makedirs(os.path.join(output_dir, 'ccgan_images'), exist_ok=True)
generate_and_save_final_images(generator, classifier, device, output_dir)