# import os
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from sklearn.model_selection import train_test_split

# #  CONFIG 
# DATASET_DIR = "dataset/Fetal/fetus_segmentation"
# IMAGES_DIR = os.path.join(DATASET_DIR, "images")
# MASKS_DIR = os.path.join(DATASET_DIR, "masks")
# SAVE_PATH = "models/fetus_unet.pth"
# EPOCHS = 8
# BATCH_SIZE = 4
# LR = 1e-4
# DEVICE = torch.device("cpu")

# #  DATASET 
# class FetusSegmentationDataset(Dataset):
#     def __init__(self, image_paths, mask_paths, transform=None):
#         self.image_paths = image_paths
#         self.mask_paths = mask_paths
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         image = Image.open(self.image_paths[idx]).convert("RGB")
#         mask = Image.open(self.mask_paths[idx]).convert("L")

#         if self.transform:
#             image = self.transform(image)
#             mask = transforms.ToTensor()(mask)

#         return image, mask

# #  MODEL (U-NET) 
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class UNet(nn.Module):
#     def __init__(self, n_channels=3, n_classes=1):
#         super(UNet, self).__init__()
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
#         self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
#         self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
#         self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#         self.conv1 = DoubleConv(512, 256)
#         self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#         self.conv2 = DoubleConv(256, 128)
#         self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#         self.conv3 = DoubleConv(128, 64)
#         self.outc = nn.Conv2d(64, n_classes, 1)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x = self.up1(x4)
#         x = self.conv1(torch.cat([x, x3], dim=1))
#         x = self.up2(x)
#         x = self.conv2(torch.cat([x, x2], dim=1))
#         x = self.up3(x)
#         x = self.conv3(torch.cat([x, x1], dim=1))
#         return torch.sigmoid(self.outc(x))

# #  TRAINING 
# def train_model():
#     image_paths = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)])
#     mask_paths = sorted([os.path.join(MASKS_DIR, f) for f in os.listdir(MASKS_DIR)])

#     train_imgs, val_imgs, train_masks, val_masks = train_test_split(
#         image_paths, mask_paths, test_size=0.2, random_state=42
#     )

#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

#     train_ds = FetusSegmentationDataset(train_imgs, train_masks, transform)
#     val_ds = FetusSegmentationDataset(val_imgs, val_masks, transform)

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

#     model = UNet().to(DEVICE)
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#     best_val_loss = float("inf")
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
#         for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
#             imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
#             preds = model(imgs)
#             loss = criterion(preds, masks)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()

#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for imgs, masks in val_loader:
#                 imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
#                 preds = model(imgs)
#                 loss = criterion(preds, masks)
#                 val_loss += loss.item()

#         print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             torch.save(model.state_dict(), SAVE_PATH)
#             print(f" Model saved (Val Loss: {val_loss/len(val_loader):.4f})")

#     print(" Training complete!")

# if __name__ == "__main__":
#     train_model()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

#  CONFIG
IMAGE_DIR = "dataset/Fetal/fetus_segmentation/images"
MASK_DIR = "dataset/Fetal/fetus_segmentation/masks"
SAVE_PATH = "models/fetus_segmentation_unet_cpu.pth"
EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-3
EARLY_STOPPING_PATIENCE = 3
DEVICE = torch.device("cpu")
print(f" Using device: {DEVICE}")

#  DATASET LOADING (FIXED)
image_paths = sorted([
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

mask_paths = []
for img_path in image_paths:
    base = os.path.splitext(os.path.basename(img_path))[0]
    candidates = [
        os.path.join(MASK_DIR, base + "_binary.png"),
        os.path.join(MASK_DIR, base + "_binary_binary.png"),
        os.path.join(MASK_DIR, base + ".png")
    ]
    mask_file = next((m for m in candidates if os.path.exists(m)), None)
    if mask_file:
        mask_paths.append(mask_file)
    else:
        print(f"Mask not found for: {base}")

# Ensure same length
assert len(image_paths) == len(mask_paths), f"❌ Mismatch: {len(image_paths)} images vs {len(mask_paths)} masks"
print(f" Total pairs found: {len(image_paths)}")

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

#  TRANSFORMS
train_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor()
])
mask_tfms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

#  DATASET CLASS
class SegmentationDataset(Dataset):
    def __init__(self, imgs, masks, transform_img=None, transform_mask=None):
        self.imgs = imgs
        self.masks = masks
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        if self.transform_img:
            img = self.transform_img(img)
        if self.transform_mask:
            mask = self.transform_mask(mask)
        return img, mask

train_ds = SegmentationDataset(train_imgs, train_masks, train_tfms, mask_tfms)
val_ds = SegmentationDataset(val_imgs, val_masks, val_tfms := mask_tfms, mask_tfms)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE)

#  SIMPLE LIGHTWEIGHT UNET
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)

        self.pool = nn.MaxPool2d(2)
        self.up1 = self.up_block(128, 64)
        self.up2 = self.up_block(64, 32)
        self.final = nn.Conv2d(32, 1, 1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def up_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 2, stride=2),
            nn.ReLU(inplace=True),
            self.conv_block(out_c, out_c)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d1 = self.up1(e3)
        d2 = self.up2(d1)
        return torch.sigmoid(self.final(d2))

model = UNet().to(DEVICE)

#  TRAINING LOOP
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_iou = 0
patience_counter = 0

def iou(pred, mask, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

print("\n Training started...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_iou = 0, 0
    for imgs, masks in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iou += iou(preds, masks).item()

    model.eval()
    val_loss, val_iou = 0, 0
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += criterion(preds, masks).item()
            val_iou += iou(preds, masks).item()

    avg_train_iou = train_iou / len(train_dl)
    avg_val_iou = val_iou / len(val_dl)
    print(f"Epoch [{epoch}/{EPOCHS}] | Train IoU: {avg_train_iou:.4f} | Val IoU: {avg_val_iou:.4f}")

    # Save best
    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        torch.save(model.state_dict(), SAVE_PATH)
        print(f" Model saved (Best Val IoU: {best_val_iou:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(" Early stopping triggered.")
            break

print(f"\n Training complete! Best Val IoU: {best_val_iou:.4f}")
