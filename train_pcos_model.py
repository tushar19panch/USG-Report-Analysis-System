import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm

# CONFIG
CSV_PATH = "dataset/csv_files/pcos_dataset.csv"
MODEL_SAVE_PATH = "models/pcos_resnet.pth"
BATCH_SIZE = 16
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 1e-4
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DATASET CLASS
class PCOSDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        print(self.data.head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        label = self.data.iloc[idx]['label']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# DATA TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# LOAD DATA
data = pd.read_csv(CSV_PATH)
classes = sorted(data['label'].unique())
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
print(f"Classes found: {class_to_idx}")

data['label'] = data['label'].map(class_to_idx)
train_data = data.sample(frac=0.8, random_state=42)
val_data = data.drop(train_data.index)

train_data.to_csv("dataset/csv_files/pcos_train_split.csv", index=False)
val_data.to_csv("dataset/csv_files/pcos_val_split.csv", index=False)

train_dataset = PCOSDataset("dataset/csv_files/pcos_train_split.csv", transform=transform)
val_dataset = PCOSDataset("dataset/csv_files/pcos_val_split.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# MODEL
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TRAINING LOOP
best_val_acc = 0
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # Early Stopping + Save Best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f" Model saved (Best Val Acc: {best_val_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(" Early stopping triggered.")
            break

print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
