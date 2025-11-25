#!/usr/bin/env python3
"""
AutoVI Dataset Federated Split - Centralized Test Data
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict
import numpy as np

def create_autovi_federated_splits_5clients(
    dataset_path: str,
    output_path: str,
    seed: int = 42
):
    random.seed(seed)
    np.random.seed(seed)
    
    print("\n" + "="*70)
    print("AutoVI FEDERATED SPLIT - 5 CLIENTS (CENTRALIZED TEST)")
    print("="*70)
    print(f"Input:  {dataset_path}")
    print(f"Output: {output_path}")
    print(f"Seed:   {seed}")
    print("="*70 + "\n")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    
    client_assignments = {
        'client_0': {
            'name': 'High-anomaly specialist',
            'categories': ['engine_wiring', 'underbody_pipes'],
            'ratio': 0.7,
            'expected_anomaly_rate': 53.1
        },
        'client_1': {
            'name': 'Medium-anomaly specialist',
            'categories': ['pipe_clip', 'pipe_staple'],
            'ratio': 0.7,
            'expected_anomaly_rate': 40.2
        },
        'client_2': {
            'name': 'Low-anomaly specialist',
            'categories': ['tank_screw'],
            'ratio': 0.7,
            'expected_anomaly_rate': 23.0
        },
        'client_3': {
            'name': 'Ultra-low anomaly specialist',
            'categories': ['underbody_screw'],
            'ratio': 0.7,
            'expected_anomaly_rate': 4.6
        },
        'client_4': {
            'name': 'Mixed portfolio',
            'categories': ['engine_wiring', 'underbody_pipes', 'pipe_clip', 
                          'pipe_staple', 'tank_screw', 'underbody_screw'],
            'ratio': 1.0,
            'expected_anomaly_rate': 35.0
        }
    }
    
    all_categories = ['engine_wiring', 'pipe_clip', 'pipe_staple', 
                     'tank_screw', 'underbody_pipes', 'underbody_screw']
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    test_data_path = os.path.join(output_path, 'test_data_centralized')
    Path(test_data_path).mkdir(parents=True, exist_ok=True)
    
    config = {
        'num_clients': 5,
        'strategy': 'anomaly_rate_based',
        'seed': seed,
        'test_data': 'centralized',
        'test_data_location': 'test_data_centralized/',
        'client_assignments': client_assignments,
        'description': 'Non-IID split: Clients 0-3 get 70% of their categories, Client 4 gets remaining 30% from all. Test data is centralized.'
    }
    
    config_path = os.path.join(output_path, 'split_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved: {config_path}\n")
    
    stats = {f'client_{i}': {} for i in range(5)}
    
    anomaly_rates = {
        'engine_wiring': 53.0,
        'underbody_pipes': 53.3,
        'pipe_clip': 42.1,
        'pipe_staple': 38.4,
        'tank_screw': 23.0,
        'underbody_screw': 4.6
    }
    
    for category in all_categories:
        print(f"{'='*70}")
        print(f"Processing: {category}")
        print(f"{'='*70}")
        
        train_path = os.path.join(dataset_path, category, category, 'train', 'good')
        
        if not os.path.exists(train_path):
            print(f"WARNING: {train_path} does not exist! Skipping...")
            continue
        
        all_images = sorted([f for f in os.listdir(train_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(all_images) == 0:
            print(f"WARNING: No images found! Skipping...")
            continue
        
        random.shuffle(all_images)
        total_images = len(all_images)
        print(f"Total images found: {total_images}")
        
        category_pool = all_images.copy()
        
        for client_id in range(4):
            client_key = f'client_{client_id}'
            assignment = client_assignments[client_key]
            
            if category not in assignment['categories']:
                stats[client_key][category] = 0
                continue
            
            ratio = assignment['ratio']
            num_images = int(total_images * ratio)
            
            client_images = category_pool[:num_images]
            category_pool = category_pool[num_images:]
            
            client_train_dir = os.path.join(
                output_path, client_key, category, 'train', 'good'
            )
            Path(client_train_dir).mkdir(parents=True, exist_ok=True)
            
            for img in client_images:
                src = os.path.join(train_path, img)
                dst = os.path.join(client_train_dir, img)
                shutil.copy2(src, dst)
            
            stats[client_key][category] = len(client_images)
            print(f"  {assignment['name']:30s}: {len(client_images):4d} images")
        
        client_key = 'client_4'
        assignment = client_assignments[client_key]
        
        if len(category_pool) > 0:
            client_train_dir = os.path.join(
                output_path, client_key, category, 'train', 'good'
            )
            Path(client_train_dir).mkdir(parents=True, exist_ok=True)
            
            for img in category_pool:
                src = os.path.join(train_path, img)
                dst = os.path.join(client_train_dir, img)
                shutil.copy2(src, dst)
            
            stats[client_key][category] = len(category_pool)
            print(f"  {assignment['name']:30s}: {len(category_pool):4d} images (remaining)")
        else:
            stats[client_key][category] = 0
        
        print()
    
    print(f"{'='*70}")
    print("Copying test data to CENTRALIZED location...")
    print(f"{'='*70}")
    
    for category in all_categories:
        test_path = os.path.join(dataset_path, category, category, 'test')
        ground_truth_path = os.path.join(dataset_path, category, category, 'ground_truth')
        
        if not os.path.exists(test_path):
            print(f"WARNING: {category} test path does not exist, skipping...")
            continue
        
        central_test_dir = os.path.join(test_data_path, category, 'test')
        shutil.copytree(test_path, central_test_dir, dirs_exist_ok=True)
        
        if os.path.exists(ground_truth_path):
            central_gt_dir = os.path.join(test_data_path, category, 'ground_truth')
            shutil.copytree(ground_truth_path, central_gt_dir, dirs_exist_ok=True)
        
        print(f"  {category}")
    
    print()
    
    for client_id in range(5):
        client_key = f'client_{client_id}'
        total_train = sum(v for k, v in stats[client_key].items() if k in all_categories)
        stats[client_key]['total_train'] = total_train
        
        weighted_anomaly = 0
        for cat in all_categories:
            count = stats[client_key].get(cat, 0)
            if count > 0:
                weighted_anomaly += count * anomaly_rates[cat]
        
        avg_anomaly_rate = weighted_anomaly / total_train if total_train > 0 else 0
        stats[client_key]['avg_anomaly_rate'] = round(avg_anomaly_rate, 2)
    
    stats_path = os.path.join(output_path, 'split_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print_statistics(stats, client_assignments, all_categories)
    
    print(f"\n{'='*70}")
    print("SPLIT COMPLETE!")
    print(f"Data saved in: {output_path}")
    print(f"Statistics: {stats_path}")
    print(f"Test data (centralized): {test_data_path}")
    print(f"{'='*70}\n")
    
    return stats


def print_statistics(stats: Dict, assignments: Dict, categories: list):
    print(f"\n{'='*70}")
    print("FEDERATED SPLIT STATISTICS")
    print(f"{'='*70}\n")
    
    for client_id in range(5):
        client_key = f'client_{client_id}'
        assignment = assignments[client_key]
        
        print(f"CLIENT {client_id}: {assignment['name']}")
        print(f"{'-'*70}")
        
        for cat in categories:
            count = stats[client_key].get(cat, 0)
            if count > 0:
                print(f"  {cat:20s}: {count:4d} images")
        
        total = stats[client_key]['total_train']
        avg_rate = stats[client_key]['avg_anomaly_rate']
        print(f"  {'-'*66}")
        print(f"  {'TOTAL':20s}: {total:4d} images (avg anomaly: {avg_rate}%)")
        print()


if __name__ == "__main__":
    
    DATASET_PATH = "/mnt/c/Users/neagu/Downloads/AutoVI_Dataset"
    OUTPUT_PATH = os.path.expanduser("~/AutoVI_federated_5clients")
    
    if os.path.exists(OUTPUT_PATH):
        print(f"Deleting old output: {OUTPUT_PATH}")
        shutil.rmtree(OUTPUT_PATH)
    
    stats = create_autovi_federated_splits_5clients(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        seed=42
    )
    
    print("Dataset split complete for Federated Learning.")
