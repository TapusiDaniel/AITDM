import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Dict, Optional, Tuple, List, Union

# UNet Architecture (Federated Model)
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """UNet architecture for federated model."""
    
    def __init__(self, n_channels: int = 3, n_classes: int = 3, bilinear: bool = True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

# UNetAutoencoder Architecture (Centralized Model)
class DoubleConvSimple(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetAutoencoder(nn.Module):
    """UNet-based autoencoder for centralized model."""
    
    def __init__(self, img_size: int = 256):
        super().__init__()
        
        self.enc1 = DoubleConvSimple(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConvSimple(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConvSimple(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConvSimple(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConvSimple(512, 1024)
        
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec1 = DoubleConvSimple(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec2 = DoubleConvSimple(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConvSimple(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = DoubleConvSimple(128, 64)
        
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))
        
        return self.out(d4)

# Dataset
class AutoVITestDataset(Dataset):
    """
    Dataset for AutoVI test data.
    
    Expected structure:
        root_dir/category/test/defect_type/*.png
    """
    
    def __init__(
        self, 
        root_dir: Union[str, Path], 
        transform: Optional[transforms.Compose] = None, 
        target_size: Tuple[int, int] = (256, 256)
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size
        self.samples: List[Dict] = []
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        self._scan_dataset()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.root_dir}")
        
        print(f"Loaded {len(self.samples)} test samples")
    
    def _scan_dataset(self) -> None:
        for category_dir in sorted(self.root_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            test_dir = category_dir / 'test'
            
            if not test_dir.exists():
                continue
            
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                
                defect_type = defect_dir.name
                is_good = (defect_type.lower() == 'good')
                
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    for img_path in sorted(defect_dir.glob(ext)):
                        self.samples.append({
                            'image_path': str(img_path),
                            'category': category_name,
                            'defect_type': defect_type,
                            'label': 0 if is_good else 1
                        })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        image = image.resize(self.target_size, Image.BILINEAR)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        
        return {
            'image': image,
            'label': sample['label'],
            'category': sample['category'],
            'defect_type': sample['defect_type'],
            'image_path': sample['image_path']
        }
    
    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        stats = {
            'total_samples': len(self.samples),
            'categories': {},
            'labels': {'good': 0, 'anomaly': 0}
        }
        
        for sample in self.samples:
            cat = sample['category']
            if cat not in stats['categories']:
                stats['categories'][cat] = {'good': 0, 'anomaly': 0, 'defect_types': set()}
            
            if sample['label'] == 0:
                stats['categories'][cat]['good'] += 1
                stats['labels']['good'] += 1
            else:
                stats['categories'][cat]['anomaly'] += 1
                stats['labels']['anomaly'] += 1
                stats['categories'][cat]['defect_types'].add(sample['defect_type'])
        
        for cat in stats['categories']:
            stats['categories'][cat]['defect_types'] = list(stats['categories'][cat]['defect_types'])
        
        return stats

# Model Loading and Inference
def detect_model_type(checkpoint: Dict) -> str:
    """Detect model architecture from checkpoint."""
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    if 'enc1.conv.0.weight' in state_dict:
        return 'centralized'
    elif 'inc.double_conv.0.weight' in state_dict:
        return 'federated'
    else:
        sample_keys = list(state_dict.keys())[:5]
        raise ValueError(f"Unknown architecture. Sample keys: {sample_keys}")


def load_model(
    checkpoint_path: Union[str, Path], 
    device: torch.device,
    model_type: Optional[str] = None
) -> nn.Module:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pth file
        device: torch device
        model_type: 'centralized' or 'federated' (auto-detected if None)
    
    Returns:
        Model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type is None:
        model_type = detect_model_type(checkpoint)
        print(f"Detected model type: {model_type}")
    
    if model_type == 'centralized':
        model = UNetAutoencoder(img_size=256)
    elif model_type == 'federated':
        model = UNet(n_channels=3, n_classes=3, bilinear=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {model_type} model ({n_params:,} parameters)")
    
    return model


def compute_anomaly_score(
    model: nn.Module, 
    image: torch.Tensor, 
    device: torch.device
) -> float:
    """Compute MSE-based anomaly score."""
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        reconstruction = model(image_batch)
        score = ((image_batch - reconstruction) ** 2).mean().item()
    return score


def compute_anomaly_map(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """Compute pixel-wise anomaly map."""
    model.eval()
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        reconstruction = model(image_batch)
        anomaly_map = ((image_batch - reconstruction) ** 2).mean(dim=1)
        anomaly_map = anomaly_map.squeeze().cpu().numpy()
    return anomaly_map


def get_default_transform() -> transforms.Compose:
    """Return default image transform."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])