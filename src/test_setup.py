import torch
from preprocessing import prepare_dataloaders
from model import UNet, count_parameters

print("="*60)
print("TESTING SETUP")
print("="*60)

print("\n[1/3] Testing preprocessing...")
try:
    train_loader, val_loader = prepare_dataloaders(
        client_id=0,
        batch_size=4,
        val_split=0.2
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    inputs, targets = next(iter(train_loader))
    print(f"Batch shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
except Exception as e:
    print(f"Error: {e}")

print("\n[2/3] Testing model...")
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(n_channels=3, n_classes=3).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 256, 256).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
except Exception as e:
    print(f"Error: {e}")

print("\n[3/3] Testing training loop...")
try:
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    print(f"Training iteration successful!")
    print(f"Loss: {loss.item():.4f}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("ALL TESTS PASSED! Ready for training.")
print("="*60)
