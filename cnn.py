import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

CLASS_NAMES = {
    0: "Center",
    1: "Donut",
    2: "Edge-Loc",
    3: "Edge-Ring",
    4: "Loc",
    5: "Near-full",
    6: "Random",
    7: "Scratch"
}

class WaferDataset(Dataset):
    def __init__(self, wafer_maps: pd.Series, labels: pd.Series):
        self.wafer_maps = wafer_maps
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        wafer_map = np.array(self.wafer_maps.iloc[idx]).reshape(32, 32)
        wafer_map = torch.tensor(wafer_map, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(int(self.labels.iloc[idx]), dtype=torch.long)
        return wafer_map, label

class WaferCNN(nn.Module):
    def __init__(self):
        super(WaferCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Input: 1 x 32 x 32 (grayscale)
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),  # First conv layer
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 16 x 16
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),  # Second conv layer
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Output: 128 x 8 x 8
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),  # First fully connected layer
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),  # Second fully connected layer
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 9)  # Output layer (8 classes + 1 normal class)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = self.classifier(x)
        return torch.softmax(x, dim=1)  # Use softmax for multi-class classification

def prepare_data(X_train, y_train, X_test, y_test):
    train_dataset = WaferDataset(X_train, y_train)
    test_dataset = WaferDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for wafers, labels in train_loader:
        wafers, labels = wafers.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(wafers)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    return train_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for wafers, labels in test_loader:
            wafers, labels = wafers.to(device), labels.to(device)
            outputs = model(wafers)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)

    # Create a dictionary of class accuracies with class names
    class_accuracies_dict = {CLASS_NAMES.get(i, f"Unknown-{i}"): acc for i, acc in enumerate(class_accuracies)}

    return overall_accuracy, class_accuracies_dict

def print_accuracies(overall_accuracy, class_accuracies, epoch=None):
    if epoch is not None:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Overall Test Accuracy: {overall_accuracy:.4f}")
    else:
        print(f"Final Overall Test Accuracy: {overall_accuracy:.4f}")
    
    print("Class Accuracies:")
    for i, acc in enumerate(class_accuracies):
        class_name = CLASS_NAMES.get(i, f"Unknown-{i}")
        print(f"{class_name}: {acc:.4f}")
    print()

# Define the output directory
output_dir = '/scratch/general/vast/u1475870/wafer_project/outputs/'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_pickle('/scratch/general/vast/u1475870/wafer_project/data/WM811K_training.pkl')  
test_df = pd.read_pickle('/scratch/general/vast/u1475870/wafer_project/data/WM811K_testing.pkl')  

X_train = train_df["waferMap"]
y_train = train_df["failureType"]
X_test = test_df["waferMap"]
y_test = test_df["failureType"]

train_loader, test_loader = prepare_data(X_train, y_train, X_test, y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = WaferCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
start_time = time.time()
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    overall_accuracy, class_accuracies = evaluate_model(model, test_loader, device)
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}")
    print(f"Epoch duration: {epoch_duration:.2f} seconds")
    print_accuracies(overall_accuracy, class_accuracies.values(), epoch)

# After the training loop
end_time = time.time()
total_duration = end_time - start_time
print(f"Total training time: {total_duration:.2f} seconds")

# Save the model
model_path = os.path.join(output_dir, 'wafer_cnn_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")