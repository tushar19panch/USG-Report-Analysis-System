import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np

# CONFIG
CSV_PATH = "dataset/csv_files/fetus_classification_dataset.csv"
SAVE_PATH = "models/fetus_classification_resnet50.pth"
BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-5
EARLY_STOPPING_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DATASET
class FetusDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# TRANSFORMS
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # Optionally, add GaussianBlur, ElasticTransform (with Albumentations)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# LOAD DATA
df = pd.read_csv(CSV_PATH)
class_names = sorted(df['label'].unique())
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
df['label'] = df['label'].map(class_to_idx)

# Stratified Split
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_ds = FetusDataset(train_df, transform=train_tfms)
val_ds = FetusDataset(val_df, transform=val_tfms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# MODEL (Fine-tune all layers)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(DEVICE)

# LOSS & OPTIMIZER
class_counts = df['label'].value_counts().sort_index().values
weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
weights = weights / weights.sum()
criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
optimizer = optim.AdamW(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# TRAINING LOOP (with F1 score)
from sklearn.metrics import f1_score
best_val_acc = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    model.train()
    train_correct, train_total, train_loss = 0, 0, 0.0
    train_preds, train_labels = [], []
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)
        train_preds.append(preds.cpu().numpy())
        train_labels.append(labels.cpu().numpy())
    train_acc = 100 * train_correct / train_total
    train_f1 = f1_score(np.concatenate(train_labels), np.concatenate(train_preds), average='weighted')
    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_preds.append(preds.cpu().numpy())
            val_labels.append(labels.cpu().numpy())
    val_acc = 100 * val_correct / val_total
    val_f1 = f1_score(np.concatenate(val_labels), np.concatenate(val_preds), average='weighted')
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.3f}")
    scheduler.step(val_acc)
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f" Model saved (Best Val Acc: {best_val_acc:.2f}%)")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break
print(f"\n Training complete! Best Validation Accuracy: {best_val_acc:.2f}%")
