import torch
import torch.nn as nn
from model import UNet
from preprocessing import get_transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import json
import argparse

def dice_score(pred_mask, true_mask, threshold=0.5):
    """Compute Dice Score for binary masks"""
    pred_binary = (pred_mask > threshold).astype(float)
    intersection = (pred_binary * true_mask).sum()
    union = pred_binary.sum() + true_mask.sum()
    
    if union == 0:
        return 1.0
    
    dice = (2.0 * intersection) / union
    return dice


def load_test_images_with_masks(test_dir, categories, img_size=256):
    """
    Load test images and ground truth masks
    
    Structure (AutoVI dataset):
        test_dir/category/
        ├── test/
        │   ├── good/
        │   │   └── XXXX.png (direct files)
        │   └── defect_type/
        │       └── XXXX.png (direct files)
        └── ground_truth/
            └── defect_type/
                └── XXXX/
                    └── 0000.png (mask always named 0000.png!)
    """
    _, transform = get_transforms(img_size, augment=False)
    
    images = []
    labels = []
    mask_paths = []
    image_paths = []
    
    for category in categories:
        category_test_dir = os.path.join(test_dir, category, 'test')
        category_gt_dir = os.path.join(test_dir, category, 'ground_truth')
        
        if not os.path.exists(category_test_dir):
            print(f"Warning: {category_test_dir} not found, skipping {category}")
            continue
        
        for defect_type in os.listdir(category_test_dir):
            defect_dir = os.path.join(category_test_dir, defect_type)
            
            if not os.path.isdir(defect_dir):
                continue
            
            is_good = (defect_type.lower() == 'good')
            
            for img_name in os.listdir(defect_dir):
                if not img_name.endswith(('.png', '.jpg', '.jpeg', '.PNG')):
                    continue
                
                img_path = os.path.join(defect_dir, img_name)
                
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(transform(img))
                    labels.append(0 if is_good else 1)
                    image_paths.append(img_path)
                    
                    if not is_good:
                        base_name = os.path.splitext(img_name)[0]
                        
                        mask_path = os.path.join(category_gt_dir, defect_type, base_name, '0000.png')
                        
                        if os.path.exists(mask_path):
                            mask_paths.append(mask_path)
                        else:
                            mask_dir = os.path.join(category_gt_dir, defect_type, base_name)
                            if os.path.exists(mask_dir):
                                mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
                                if mask_files:
                                    mask_path = os.path.join(mask_dir, mask_files[0])
                                    mask_paths.append(mask_path)
                                else:
                                    mask_paths.append(None)
                            else:
                                mask_paths.append(None)
                    else:
                        mask_paths.append(None)
                        
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
    
    return images, labels, mask_paths, image_paths


def compute_anomaly_scores_and_maps(model, images, device, batch_size=16):
    model.eval()
    scores = []
    anomaly_maps = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Computing anomaly scores"):
        batch = torch.stack(images[i:i+batch_size]).to(device)
        
        with torch.no_grad():
            reconstructed = model(batch)
            
            mse_per_image = torch.mean((batch - reconstructed) ** 2, dim=[1, 2, 3])
            scores.extend(mse_per_image.cpu().numpy())
            
            mse_per_pixel = torch.mean((batch - reconstructed) ** 2, dim=1)
            anomaly_maps.extend(mse_per_pixel.cpu().numpy())
    
    return np.array(scores), anomaly_maps


def load_ground_truth_mask(mask_path, img_size=256):
    if mask_path is None or not os.path.exists(mask_path):
        return None
    
    try:
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((img_size, img_size), Image.NEAREST)
        mask = np.array(mask)
        binary_mask = (mask > 127).astype(float)
        return binary_mask
    except Exception as e:
        print(f"⚠️ Error loading mask {mask_path}: {e}")
        return None


def evaluate_model(checkpoint_path, test_dir, categories, device='cuda', img_size=256):
    model = UNet(n_channels=3, n_classes=3).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', 'N/A')
    else:
        model.load_state_dict(checkpoint)
        epoch = 'N/A'
        val_loss = 'N/A'
    
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"Evaluating: {os.path.basename(checkpoint_path)}")
    print(f"Epoch: {epoch}, Val Loss: {val_loss}")
    print(f"Categories: {categories}")
    print(f"{'='*70}\n")
    
    print(f"Loading test images from {test_dir}...")
    images, labels, mask_paths, image_paths = load_test_images_with_masks(
        test_dir, categories, img_size
    )
    
    if len(images) == 0:
        print("ERROR: No images loaded! Check test_dir and categories.")
        return None, None, None
    
    print(f"Loaded {len(images)} images ({sum(labels)} defects, {len(labels)-sum(labels)} good)")
    
    print("\nComputing anomaly scores and maps...")
    scores, anomaly_maps = compute_anomaly_scores_and_maps(model, images, device)
    
    if len(set(labels)) > 1:
        auroc = roc_auc_score(labels, scores)
    else:
        auroc = 0.0
        print("Warning: Only one class present, AUROC not computable")
    
    dice_scores = []
    valid_masks = 0
    
    print("\nComputing Dice scores...")
    for i, (mask_path, anomaly_map) in enumerate(zip(mask_paths, anomaly_maps)):
        if labels[i] == 1:
            gt_mask = load_ground_truth_mask(mask_path, img_size)
            if gt_mask is not None:
                threshold = np.percentile(anomaly_map, 90)
                dice = dice_score(anomaly_map, gt_mask, threshold)
                dice_scores.append(dice)
                valid_masks += 1
    
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    results = {
        'checkpoint': os.path.basename(checkpoint_path),
        'categories': categories,
        'auroc': float(auroc),
        'dice_score': float(avg_dice),
        'num_images': len(images),
        'num_defects': int(sum(labels)),
        'num_good': len(labels) - int(sum(labels)),
        'num_masks_evaluated': valid_masks,
        'mean_score_good': float(scores[np.array(labels) == 0].mean()) if sum(np.array(labels) == 0) > 0 else 0.0,
        'mean_score_defect': float(scores[np.array(labels) == 1].mean()) if sum(labels) > 0 else 0.0,
    }
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"AUROC (image-level):      {results['auroc']:.4f}")
    print(f"Dice Score (pixel-level): {results['dice_score']:.4f}")
    print(f"\nDataset Statistics:")
    print(f"  Total images:     {results['num_images']}")
    print(f"  Good images:      {results['num_good']}")
    print(f"  Defect images:    {results['num_defects']}")
    print(f"  Masks evaluated:  {results['num_masks_evaluated']}")
    print(f"\nAnomaly Scores:")
    print(f"  Good images:   {results['mean_score_good']:.6f}")
    print(f"  Defect images: {results['mean_score_defect']:.6f}")
    print(f"  Separation:    {results['mean_score_defect'] - results['mean_score_good']:.6f}")
    print(f"{'='*70}\n")
    
    return results, scores, anomaly_maps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on test data')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, 
                        default='data/test_data_centralized',
                        help='Test data directory')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                        help='Categories to evaluate (default: all)')
    parser.add_argument('--client_id', type=int, default=None,
                        help='Client ID (to load categories from config)')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.categories is not None:
        categories = args.categories
    elif args.client_id is not None:
        with open('data/federated_data/split_config.json') as f:
            config = json.load(f)
        
        if 'clients' in config:
            categories = config['clients'][f'client_{args.client_id}']['categories']
        elif 'client_assignments' in config:
            categories = config['client_assignments'][f'client_{args.client_id}']['categories']
        else:
            raise ValueError("Invalid config format")
    else:
        categories = [d for d in os.listdir(args.test_dir) 
                     if os.path.isdir(os.path.join(args.test_dir, d))]
        categories.sort()
    
    print(f"Evaluating on categories: {categories}\n")
    
    results, scores, anomaly_maps = evaluate_model(
        args.checkpoint, 
        args.test_dir, 
        categories, 
        device,
        args.img_size
    )
    
    if results is None:
        print("Evaluation failed!")
        exit(1)
    
    os.makedirs(args.output, exist_ok=True)
    
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    results_file = os.path.join(args.output, f'{checkpoint_name}_evaluation.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
