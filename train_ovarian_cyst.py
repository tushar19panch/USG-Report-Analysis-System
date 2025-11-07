import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision.models.segmentation import fcn_resnet50

# CONFIG
BASE_DIR = "dataset/Pelvis/ovarian_cyst"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")
MODEL_SAVE_PATH = "models/ovarian_cyst_unet.pth"

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# DATASET
class OvarianCystDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)

        mask = transforms.Resize((IMG_SIZE, IMG_SIZE))(mask)
        mask = transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()  # Binary mask

        return image, mask

# TRANSFORMS
train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3-channel normalization
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# FUNCTION TO MATCH MASK TO IMAGE
def get_mask_for_image(img_name, mask_dir):
    base_name = os.path.splitext(img_name)[0]  # e.g., '1' from '1.JPG'
    for mask_file in os.listdir(mask_dir):
        if mask_file.startswith(base_name):
            return os.path.join(mask_dir, mask_file)
    return None

# GATHER IMAGE FILES
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
mask_files_matched = []

# MATCH MASKS TO IMAGES
full_image_paths = []
full_mask_paths = []

for img_name in image_files:
    mask_path = get_mask_for_image(img_name, MASK_DIR)
    if mask_path is None:
        raise ValueError(f"No matching mask found for image {img_name}")
    full_image_paths.append(os.path.join(IMAGE_DIR, img_name))
    full_mask_paths.append(mask_path)

print(f"Found {len(full_image_paths)} images and {len(full_mask_paths)} masks matched.")

# SPLIT DATASET
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    full_image_paths, full_mask_paths, test_size=0.2, random_state=42
)

train_ds = OvarianCystDataset(train_imgs, train_masks, transform=train_tfms)
val_ds = OvarianCystDataset(val_imgs, val_masks, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total pairs: {len(train_ds)+len(val_ds)} | Train: {len(train_ds)} | Val: {len(val_ds)}")

# MODEL
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = fcn_resnet50(pretrained=True)
        self.encoder.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        out = self.encoder(x)['out']
        return torch.sigmoid(out)

model = SimpleUNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# IOU METRIC
def iou_score(pred, target):
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# TRAINING LOOP
best_iou = 0
patience = 0

print("\nTraining started...\n")
for epoch in range(EPOCHS):
    model.train()
    train_iou = []
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_iou.append(iou_score(preds, masks).item())

    model.eval()
    val_iou = []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            val_iou.append(iou_score(preds, masks).item())

    avg_train_iou = np.mean(train_iou)
    avg_val_iou = np.mean(val_iou)
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f}")

    # Save best model
    if avg_val_iou > best_iou:
        best_iou = avg_val_iou
        patience = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved (Best IoU: {best_iou:.4f})")
    else:
        patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print(f"\nTraining complete! Best validation IoU: {best_iou:.4f}")
