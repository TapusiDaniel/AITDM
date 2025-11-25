import torch
import torch.nn as nn
from torch.optim import Adam
from model import UNet, count_parameters
from preprocessing import prepare_dataloaders
import os
from tqdm import tqdm

def train_client_standalone(client_id, epochs=50, lr=0.001, device='cuda', save_dir='checkpoints/standalone'):
    """
    Train UNet on a single client without federated setup
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training Client {client_id}")
    print(f"{'='*60}\n")
    
    train_loader, val_loader = prepare_dataloaders(
        client_id=client_id,
        batch_size=16,
        val_split=0.2,
        img_size=256,
        augment=True
    )
    
    model = UNet(n_channels=3, n_classes=3).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  LR: {new_lr:.6f}", end='')
        if new_lr < old_lr:
            print(f" (reduced from {old_lr:.6f})")
        else:
            print()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, f'client_{client_id}_unet_best.pth'))
            print(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        
        print(f"{'-'*60}")
    
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }, os.path.join(save_dir, f'client_{client_id}_unet_final.pth'))
    
    print(f"\n{'='*60}")
    print(f"Training complete for Client {client_id}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model, train_losses, val_losses = train_client_standalone(
        client_id=4,
        epochs=10,
        lr=0.001,
        device=device
    )
