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
    5: "Random",
    6: "Scratch",
    7: "Near-full"
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

class WaferCNN(nn.Module):
    def __init__(self, num_classes):
        super(WaferCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.softmax(x, dim=1)

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

    class_accuracies_dict = {CLASS_NAMES.get(i, f"Unknown-{i}"): acc for i, acc in enumerate(class_accuracies)}

    return overall_accuracy, class_accuracies_dict

def print_accuracies(overall_accuracy, class_accuracies, epoch=None):
    if epoch is not None:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Overall Test Accuracy: {overall_accuracy:.4f}")
    else:
        print(f"Final Overall Test Accuracy: {overall_accuracy:.4f}")
    
    print("Class Accuracies:")
    for class_name, acc in class_accuracies.items():
        print(f"{class_name}: {acc:.4f}")
    print()

# Define the output directory
output_dir = '/scratch/general/vast/u1475870/wafer_project/outputs/'

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
train_df = pd.read_pickle('/scratch/general/vast/u1475870/wafer_project/data/WM811K_training.pkl')  
test_df = pd.read_pickle('/scratch/general/vast/u1475870/wafer_project/data/WM811K_testing.pkl')  

X_train = train_df["waferMap"].tolist()
y_train = train_df["failureType_list"].apply(lambda x: x.index(1)).tolist()
X_test = test_df["waferMap"].tolist()
y_test = test_df["failureType_list"].apply(lambda x: x.index(1)).tolist()

train_loader, test_loader = prepare_data(X_train, y_train, X_test, y_test)

num_classes = len(CLASS_NAMES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = WaferCNN(num_classes).to(device)
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
    print_accuracies(overall_accuracy, class_accuracies, epoch)

# After the training loop
end_time = time.time()
total_duration = end_time - start_time
print(f"Total training time: {total_duration:.2f} seconds")

# Save the model
model_path = os.path.join(output_dir, 'wafer_cnn_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")