import flwr as fl
from flwr.common import parameters_to_ndarrays
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import OrderedDict
from model import UNet, count_parameters
from preprocessing import prepare_dataloaders
import argparse
import os
from datetime import datetime

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, train_loader, val_loader, device):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.MSELoss()
    
    def get_parameters(self, config):
        """Return model parameters as NumPy arrays"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from NumPy arrays"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train model locally for one federated round"""
        self.set_parameters(parameters)
        
        epochs = config.get("local_epochs", 2)
        lr = config.get("lr", 0.001)
        
        optimizer = Adam(self.model.parameters(), lr=lr)
        self.model.train()
        
        total_loss = 0.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                del inputs, targets, outputs, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            avg_loss = epoch_loss / batch_count
            total_loss += avg_loss
            print(f"  Client {self.client_id} - Local Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        avg_total_loss = total_loss / epochs
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "client_id": self.client_id,
            "train_loss": avg_total_loss
        }
    
    def evaluate(self, parameters, config):
        """Evaluate model locally"""
        self.set_parameters(parameters)
        
        self.model.eval()
        val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                batch_count += 1
                
                del inputs, targets, outputs, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / batch_count
        print(f"  Client {self.client_id} - Validation Loss: {avg_val_loss:.4f}")
        
        return float(avg_val_loss), len(self.val_loader.dataset), {
            "client_id": self.client_id,
            "val_loss": avg_val_loss
        }


# Global device variable for client_fn
DEVICE = None

def client_fn(cid: str):
    """Create a Flower client for given client ID"""
    global DEVICE
    client_id = int(cid)
    
    print(f"\n{'='*50}")
    print(f"Initializing Client {client_id} on {DEVICE.type.upper()}")
    print(f"{'='*50}")
    
    train_loader, val_loader = prepare_dataloaders(
        client_id=client_id,
        batch_size=2, 
        val_split=0.2,
        img_size=256,
        augment=True
    )
    
    model = UNet(n_channels=3, n_classes=3).to(DEVICE)
    
    return FlowerClient(client_id, model, train_loader, val_loader, DEVICE)


def get_evaluate_fn(device):
    """Return evaluation function for server-side evaluation"""
    
    def evaluate(server_round, parameters, config):
        model = UNet(n_channels=3, n_classes=3).to(device)
        
        if isinstance(parameters, fl.common.Parameters):
            ndarrays = parameters_to_ndarrays(parameters)
        else:
            ndarrays = parameters
        
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        
        _, val_loader = prepare_dataloaders(
            client_id=0,
            batch_size=2,
            val_split=0.2,
            img_size=256,
            augment=False
        )
        
        model.eval()
        criterion = nn.MSELoss()
        val_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                batch_count += 1
                
                del inputs, targets, outputs, loss
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        avg_val_loss = val_loss / batch_count
        
        print(f"\n{'='*60}")
        print(f"Round {server_round} - Global Model Validation")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return avg_val_loss, {"val_loss": avg_val_loss}
    
    return evaluate


def save_global_model(parameters, round_num, save_dir='checkpoints/federated'):
    """Save global model after federated round"""
    os.makedirs(save_dir, exist_ok=True)
    
    model = UNet(n_channels=3, n_classes=3)
    
    if isinstance(parameters, fl.common.Parameters):
        ndarrays = parameters_to_ndarrays(parameters)
    else:
        ndarrays = parameters
    
    params_dict = zip(model.state_dict().keys(), ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    save_path = os.path.join(save_dir, f'global_model_round_{round_num}.pth')
    
    torch.save({
        'round': round_num,
        'model_state_dict': model.state_dict(),
        'strategy': 'FedAvg',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }, save_path)
    
    print(f"✓ Global model saved: {save_path}")
    return save_path


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that saves global model after each round"""
    
    def __init__(self, save_dir='checkpoints/federated', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            save_global_model(aggregated_parameters, server_round, self.save_dir)
        
        return aggregated_parameters, aggregated_metrics


def main():
    global DEVICE
    
    parser = argparse.ArgumentParser(description='Federated Learning with Flower')
    parser.add_argument('--num_rounds', type=int, default=5, 
                        help='Number of federated rounds')
    parser.add_argument('--num_clients', type=int, default=5, 
                        help='Total number of clients')
    parser.add_argument('--clients_per_round', type=int, default=1,
                        help='Number of clients per round')
    parser.add_argument('--local_epochs', type=int, default=3,
                        help='Local epochs per round')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='checkpoints/federated',
                        help='Directory to save global models')
    args = parser.parse_args()
    
    if args.device == 'auto':
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        DEVICE = torch.device(args.device)
    
    if DEVICE.type == 'cuda':
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
    
    print(f"\n{'='*60}")
    print(f"Federated Learning with Flower")
    print(f"{'='*60}")
    print(f"Device: {DEVICE.type.upper()}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Rounds: {args.num_rounds}")
    print(f"Total clients: {args.num_clients}")
    print(f"Clients per round: {args.clients_per_round}")
    print(f"Local epochs per round: {args.local_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Save directory: {args.save_dir}")
    print(f"{'='*60}\n")
    
    initial_model = UNet(n_channels=3, n_classes=3)
    print(f"Model parameters: {count_parameters(initial_model):,}\n")
    
    import ray
    
    if DEVICE.type == 'cuda':
        num_gpus = torch.cuda.device_count()
        print(f"Detected {num_gpus} GPU(s)")
        
        gpu_per_client = 0.5
        
        ray.init(
            num_cpus=4,
            num_gpus=num_gpus,
            _memory=8 * 1024 * 1024 * 1024,
            object_store_memory=1 * 1024 * 1024 * 1024,
            ignore_reinit_error=True
        )
        
        client_resources = {
            "num_cpus": 2,
            "num_gpus": gpu_per_client
        }
        
        ray_init_args = {
            "num_cpus": 4,
            "num_gpus": num_gpus,
            "include_dashboard": False
        }
    else:
        print("Running on CPU (slower)")
        
        ray.init(
            num_cpus=1,
            _memory=5 * 1024 * 1024 * 1024,
            object_store_memory=500 * 1024 * 1024,
            ignore_reinit_error=True
        )
        
        client_resources = {
            "num_cpus": 1,
            "num_gpus": 0.0
        }
        
        ray_init_args = {
            "num_cpus": 1,
            "include_dashboard": False
        }
    
    fraction = args.clients_per_round / args.num_clients
    
    strategy = SaveModelStrategy(
        save_dir=args.save_dir,
        fraction_fit=fraction,
        fraction_evaluate=fraction,
        min_fit_clients=args.clients_per_round,
        min_evaluate_clients=args.clients_per_round,
        min_available_clients=args.num_clients,
        evaluate_fn=get_evaluate_fn(DEVICE),
        on_fit_config_fn=lambda round: {
            "local_epochs": args.local_epochs,
            "lr": args.lr
        }
    )
    
    print(f"{'='*60}")
    print(f"Starting Federated Training")
    print(f"{'='*60}\n")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        client_resources=client_resources
    )
    
    print(f"\n{'='*60}")
    print(f"Federated Training Completed!")
    print(f"{'='*60}")
    print(f"Total rounds: {args.num_rounds}")
    print(f"Global models saved in: {args.save_dir}/")
    print(f"\nSaved models:")
    
    for round_num in range(1, args.num_rounds + 1):
        model_path = os.path.join(args.save_dir, f'global_model_round_{round_num}.pth')
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"  - Round {round_num}: {model_path} ({size_mb:.2f} MB)")
    
    print(f"{'='*60}\n")
    
    ray.shutdown()
    
    return history


if __name__ == "__main__":
    main()
