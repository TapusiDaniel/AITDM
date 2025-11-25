# import flwr as fl
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2 as transforms

from torch.utils.data import random_split

import os
import pandas as pd
from PIL import Image

import matplotlib.pyplot as plt

from tqdm import tqdm

from model import UNetAutoencoder

# -------------------------------
# Dataset class
# -------------------------------
class AutoVIDataset(torch.utils.data.Dataset):
    def __init__(self, device, path=None, transform=None):
        if path is None:
            data_path = [f"Dataset/client_{i}" for i in range(5)]
        else:
            data_path = path

        if isinstance(data_path, str):
            data_path = [data_path]

        paths = []

        for client_path in data_path:
            for subdir, _, files in os.walk(client_path):
                if files:
                    for file in files:
                        paths.append(os.path.join(subdir, file))

        self.df = pd.DataFrame(paths, columns=['path'])
        self.transform = transform

        self.device = device
    
    def __getitem__(self, index):
        img_path = self.df.iloc[index]['path']
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        image = image.to(self.device)

        return image, image
    
    def __len__(self):
        return len(self.df)

# -------------------------------
# Save Plots
# -------------------------------
def plot_and_save_losses(train_losses, val_losses, save_path="loss_plot.png"):
    """
    Plot and save training and validation losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to {save_path}")    

# -------------------------------
# Evaluation Loop
# -------------------------------
def evaluate(model, dataloader, device, criterion, split_name="Valid"):
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            total += labels.size(0)
    avg_loss = total_loss / total
    print(f"Evaluating... - {split_name} Loss: {avg_loss:.4f}, {split_name}")
    return avg_loss


# -------------------------------
# Train Loop
# -------------------------------
def train(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        iterations = 0
        for batch_images, batch_labels in tqdm(train_dataloader):
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iterations += 1

        train_epoch_loss = epoch_loss / iterations
        print(f"Epoch {epoch+1}, Loss: {train_epoch_loss:.4f}")
        train_losses.append(train_epoch_loss)

        eval_epoch_loss = evaluate(model, val_dataloader, device, criterion, split_name="Valid")
        val_losses.append(eval_epoch_loss)

        if eval_epoch_loss < best_val_loss:
            best_val_loss = eval_epoch_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_epoch_loss, 'val_loss': eval_epoch_loss,
                        }, os.path.join("checkpoints", f'model_best_loss.pth'))
            print(f"Saved model (val_loss: {best_val_loss:.4f})")
        
        scheduler.step(eval_epoch_loss)
    
    torch.save({'epoch': epochs, 'model_state_dict': model.state_dict(), 'train_losses': train_losses, 'val_losses': val_losses,
                    'best_val_loss': best_val_loss
                }, os.path.join("final_models", f'final.pth'))

    plot_and_save_losses(train_losses, val_losses, "training_losses.png")

# -------------------------------
# Hyperparameters
# -------------------------------
BATCH_SIZE = 16
NO_EPOCHS = 10

if __name__ == "__main__":
    train_transform = torch.nn.Sequential(
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )

    valid_transform  = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")     

    full_dataset = AutoVIDataset(device=device, path=None, transform=train_transform)
    
    total_size = len(full_dataset)
    val_size = int(0.15 * total_size)
    train_size = total_size - val_size

    # Split
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Set the according transforms for validation subset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = valid_transform
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model = UNetAutoencoder()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    train(model, train_dataloader, val_dataloader, device, criterion, optimizer, scheduler, epochs=NO_EPOCHS)
