import torch
import os
from model import UNet

def aggregate_models(client_checkpoints, output_path):
    """
    Aggregate multiple client models by averaging their weights (FedAvg)
    
    Args:
        client_checkpoints: List of paths to client model checkpoints
        output_path: Path to save the aggregated model
    """
    
    print(f"\n{'='*70}")
    print("AGGREGATING STANDALONE MODELS (FedAvg)")
    print(f"{'='*70}\n")
    
    aggregated_model = UNet(n_channels=3, n_classes=3)
    aggregated_state_dict = aggregated_model.state_dict()
    
    client_state_dicts = []
    for i, checkpoint_path in enumerate(client_checkpoints):
        print(f"Loading client {i}: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"WARNING: {checkpoint_path} not found, skipping")
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        client_state_dicts.append(state_dict)
    
    if len(client_state_dicts) == 0:
        print("ERROR: No valid client models found!")
        return
    
    print(f"\nLoaded {len(client_state_dicts)} client models")
    print("\nAggregating weights (averaging)...")
    
    for key in aggregated_state_dict.keys():
        stacked_weights = torch.stack([client_sd[key].float() for client_sd in client_state_dicts])
        
        aggregated_state_dict[key] = torch.mean(stacked_weights, dim=0)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save({
        'model_state_dict': aggregated_state_dict,
        'num_clients': len(client_state_dicts),
        'client_checkpoints': client_checkpoints,
        'aggregation_method': 'FedAvg (uniform averaging)'
    }, output_path)
    
    print(f"\nAggregated model saved to: {output_path}")
    print(f"  - Number of clients: {len(client_state_dicts)}")
    print(f"  - Aggregation method: FedAvg (uniform averaging)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    client_checkpoints = [
        'checkpoints/standalone/client_0_unet_best.pth',
        'checkpoints/standalone/client_1_unet_best.pth',
        'checkpoints/standalone/client_2_unet_best.pth',
        'checkpoints/standalone/client_3_unet_best.pth',
        'checkpoints/standalone/client_4_unet_best.pth'
    ]
    
    output_path = 'checkpoints/standalone/aggregated_from_standalone.pth'
    
    aggregate_models(client_checkpoints, output_path)
