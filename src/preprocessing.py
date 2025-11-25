# preprocessing.py

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

class AutoVIDataset(Dataset):
    """
    Dataset pentru AutoVI Industrial Anomaly Detection
    Loads only 'good' images for reconstruction training
    """
    def __init__(self, root_dir, categories, transform=None):
        """
        Args:
            root_dir: path to client data (e.g., 'data/federated_data/client_0/')
            categories: list of categories (e.g., ['engine_wiring', 'underbody_pipes'])
            transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.categories = categories
        self.transform = transform
        self.images = []
        
        for category in categories:
            good_dir = os.path.join(root_dir, category, 'train', 'good')
            if os.path.exists(good_dir):
                for img_name in os.listdir(good_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(good_dir, img_name))
        
        print(f"Loaded {len(self.images)} good images from {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, image


def get_transforms(img_size=256, augment=True):
    """
    Returns train and validation transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def prepare_dataloaders(client_id, data_root='data/federated_data', 
                        batch_size=16, val_split=0.2, 
                        img_size=256, augment=True, num_workers=2):
    """
    Prepare train and validation DataLoaders for a specific client
    
    Args:
        client_id: client number (0-4)
        data_root: root directory containing federated_data/
        batch_size: batch size for DataLoader
        val_split: fraction of data for validation (0.2 = 20%)
        img_size: image size for resizing
        augment: whether to apply data augmentation
        num_workers: number of workers for DataLoader
    
    Returns:
        train_loader, val_loader
    """
    client_dir = os.path.join(data_root, f'client_{client_id}')
    
    split_config_path = os.path.join(data_root, 'split_config.json')
    with open(split_config_path, 'r') as f:
        split_config = json.load(f)
    
    if 'clients' in split_config:
        categories = split_config['clients'][f'client_{client_id}']['categories']
    elif 'client_assignments' in split_config:
        categories = split_config['client_assignments'][f'client_{client_id}']['categories']
    else:
        raise ValueError("Invalid split_config.json format. Must have 'clients' or 'client_assignments' key.")
    
    train_transform, val_transform = get_transforms(img_size, augment)
    
    full_dataset = AutoVIDataset(
        root_dir=client_dir,
        categories=categories,
        transform=train_transform
    )
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Client {client_id}: Train={train_size}, Val={val_size}")
    
    return train_loader, val_loader


def get_test_loader(test_dir='data/test_data_centralized', 
                    batch_size=16, img_size=256, num_workers=2):
    """
    Get test DataLoader for centralized evaluation
    """
    _, val_transform = get_transforms(img_size, augment=False)
    
    categories = [d for d in os.listdir(test_dir) 
                  if os.path.isdir(os.path.join(test_dir, d))]
    
    test_dataset = AutoVIDataset(
        root_dir=test_dir,
        categories=categories,
        transform=val_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader
