#!/usr/bin/env python3
"""
Adversarial emotion recognition training script
Uses high/low difficulty neutral samples as contrastive learning approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import yaml
import random
import math 
import torch.optim.lr_scheduler as lr_scheduler
from functions import *

# Add parent directory to path to access original dataset classes
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from collections import defaultdict
import wandb

# Import local modules
from config import Config
from model import *
from functions import *


class AdversarialDataLoader:
    """Custom data loader for adversarial learning"""
    
    def __init__(self, dataset, indices, batch_size, difficulty_threshold=0.5):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.difficulty_threshold = difficulty_threshold
        self.num_batches = len(indices) // batch_size + (1 if len(indices) % batch_size != 0 else 0)
    
    def __iter__(self):
        # Shuffle the indices at the start of each iteration (epoch)
        random.shuffle(self.indices)

        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch_indices = self.indices[start_idx:end_idx]

            batch = {
                'features': torch.stack([self.dataset[idx]['features'] for idx in batch_indices]),
                'label': torch.stack([self.dataset[idx]['label'] for idx in batch_indices]),
                'difficulty': torch.tensor([self.dataset.data[idx]['difficulty'] for idx in batch_indices]),
                'speaker_id': [self.dataset[idx]['speaker_id'] for idx in batch_indices],
                'session': [self.dataset[idx]['session'] for idx in batch_indices],
                'dataset': [self.dataset[idx]['dataset'] for idx in batch_indices]
            }
            yield batch
    
    def __len__(self):
        return self.num_batches


# def adversarial_loss(logits, labels, difficulties, alpha=0.5):
#     """
#     Adversarial loss that penalizes easy neutral predictions and rewards difficult ones
#     alpha: weight for adversarial component
#     """
#     # Standard cross-entropy loss
#     ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
    
#     # Get neutral mask (class 0)
#     neutral_mask = (labels == 0).float()
    
#     # For neutral samples, weight by inverse difficulty
#     # High difficulty neutral should have lower loss (easier to learn from)
#     # Low difficulty neutral should have higher loss (harder to distinguish)
#     neutral_weights = torch.where(neutral_mask.bool(), 
#                                 1.0 + (1.0 - difficulties),  # Inverse difficulty for neutral
#                                 torch.ones_like(difficulties))  # Normal weight for others
    
#     # Apply adversarial weighting
#     adversarial_loss = ce_loss * neutral_weights
#     loss = (1 - alpha) * ce_loss + alpha * adversarial_loss

    
#     return loss.mean()

def adversarial_loss(logits, embeddings, labels, difficulties, alpha=0.5,num_classes=None, device='cpu'):
    #    """
    #     Combines:
    #     - Batch-wise class weights for CE loss.
    #     - Hard-mined triplet loss with difficulty weighting for all classes.
        
    #     Hard mining:
    #     - Positive: same class, farthest from anchor in embedding space.
    #     - Negative: different class, closest to anchor.
        
    #     High difficulty = harder sample → bigger weight.
    #     """
    margin = 1.0
    device = logits.device  # ensure everything matches model output

    
    # labels = torch.tensor(labels, dtype=torch.long, device=device)    


    embeddings = embeddings.to(device)
    
    if num_classes is None:
        num_classes = logits.size(1)
    
    batch_size = labels.size(0)

    # === 1. Batchwise inverse-frequency weights for CE ===
    class_counts = torch.bincount(labels, minlength=num_classes).float().to(device)
    class_counts = torch.where(class_counts == 0, torch.ones_like(class_counts), class_counts)
    class_weights = (1.0 / class_counts)
    class_weights = class_weights / class_weights.max()
    
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    ce_loss = ce_loss_fn(logits, labels)  # shape: [batch_size]
    
    # === 2. Hard-mined triplets ===
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # shape: [B, B]
    
    triplet_losses = []
    for i in range(batch_size):
        anchor_label = labels[i]
        anchor_diff = difficulties[i]  # high = harder
        
        pos_mask = (labels == anchor_label) & (torch.arange(batch_size, device=device) != i)
        if pos_mask.sum() == 0:
            continue
        
        neg_mask = labels != anchor_label
        if neg_mask.sum() == 0:
            continue
        
        # Hardest positive (max distance)
        # Calculate difficulty difference with anchor for all positives
        pos_diff = torch.abs(difficulties[pos_mask] - anchor_diff)

        # Combine distance and difficulty difference (weights can be tuned)
        score_pos = dist_matrix[i][pos_mask] * pos_diff  # elementwise multiply

        # Pick positive with max combined score
        hardest_pos_idx = torch.argmax(score_pos)
        pos_indices = torch.nonzero(pos_mask, as_tuple=False).flatten()
        hardest_pos = pos_indices[hardest_pos_idx]
        
        # Hardest negative (min distance)
        neg_indices = torch.nonzero(neg_mask, as_tuple=False).flatten()
        hardest_neg_idx = torch.argmin(dist_matrix[i][neg_mask])
        hardest_neg = neg_indices[hardest_neg_idx]
        
        # Triplet loss
        triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction='none')
        loss_val = triplet_loss_fn(
            embeddings[i].unsqueeze(0),
            embeddings[hardest_pos].unsqueeze(0),
            embeddings[hardest_neg].unsqueeze(0)
        )
        
        # Weight by difficulty
        loss_val = loss_val * (1.0 + anchor_diff)
        
        triplet_losses.append(loss_val)
    
    if len(triplet_losses) > 0:
        triplet_loss_val = torch.cat(triplet_losses).mean()
    else:
        triplet_loss_val = torch.tensor(0.0, device=device)
    
    # === 3. Combine CE + Triplet ===
    total_loss = (1 - alpha) * ce_loss.mean() + alpha * triplet_loss_val
    return total_loss

def train_epoch_adversarial(model, data_loader, criterion, optimizer, scheduler, device, alpha=0.5):
    """Train model for one epoch with adversarial learning"""
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    for batch in data_loader:
        features = batch['features'].to(device)
        batch_labels = batch['label'].to(device)
        difficulties = batch['difficulty'].to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        
        # Use adversarial loss
        loss = adversarial_loss(logits, features, batch_labels, difficulties)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        neutral_mask = (batch_labels == 0)
        if neutral_mask.any():
            neutral_difficulties = difficulties[neutral_mask]
            log_dict = {
                "loss": loss,
                "lr": current_lr
            }
        else:
            log_dict = {
                "loss": loss,
                "lr": current_lr
            }
        
        wandb.log(log_dict)
        
        # Track predictions for metrics
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions.extend(preds)
        labels.extend(batch_labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(predictions, labels)
    scheduler.step()
    
    return avg_loss, metrics


def run_adversarial_cross_corpus_evaluation(config, train_dataset, test_datasets):
    """Run cross-corpus evaluation with adversarial learning"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print(f"🎭 Adversarial Learning: Enabled")
    print(f"📊 Cross-Corpus Mode: Train={1-config.val_split:.0%}, Val={config.val_split:.0%}")
    
    # Create train/validation split
    total_samples = len(train_dataset)
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    
    val_size = int(total_samples * config.val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    print(f"📈 Training samples: {len(train_indices)}")
    print(f"📋 Validation samples: {len(val_indices)}")

    # Create datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    
    # Create adversarial data loader for training
    train_loader = AdversarialDataLoader(train_dataset, train_indices, config.batch_size)
    
    # Create test loaders for cross-corpus datasets
    test_loaders = []
    test_names = []
    if isinstance(test_datasets, list):
        for test_dataset in test_datasets:
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
            test_loaders.append(test_loader)
            test_names.append(test_dataset.dataset_name)
    else:
        test_loader = DataLoader(test_datasets, batch_size=config.batch_size, shuffle=False)
        test_loaders = [test_loader]
        test_names = [test_datasets.dataset_name]
    
    print(f"🎯 Test datasets: {', '.join(test_names)}")
    
    # Get actual feature dimension from dataset
    sample_features = train_dataset[0]['features']
    if len(sample_features.shape) == 2:
        actual_input_dim = sample_features.shape[-1]
    else:
        actual_input_dim = sample_features.shape[0]
    
    print(f"🔍 Using input_dim: {actual_input_dim} (detected from {train_dataset.dataset_name})")
    
    # Initialize model
    model = SimpleEmotionClassifier(
        input_dim=actual_input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLossAutoWeights(num_classes=4, gamma=2.0, reduction='mean', device=device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    # Training loop with adversarial learning
    best_val_acc = 0
    adversarial_alpha = getattr(config, 'adversarial_alpha', 0.5)
    
    for epoch in range(config.num_epochs):
        print(f"   Epoch {epoch+1}: Adversarial training with alpha={adversarial_alpha:.2f}")
        
        train_loss, train_metrics = train_epoch_adversarial(
            model, train_loader, criterion, optimizer, scheduler, device, adversarial_alpha
        )
        
        val_results = evaluate_model(model, val_loader, criterion, device, create_plots=False)
        val_dict = {
            "val/accuracy": val_results['accuracy'],
            "val/loss": val_results['loss'],
            "val/uar": val_results['uar'],
            "val/f1": val_results['f1_weighted'],
        }
        
        wandb.log(val_dict)
        if val_results['accuracy'] > best_val_acc:
            best_val_acc = val_results['accuracy']
            best_model_state = model.state_dict().copy()
        
        print(f"   Epoch {epoch+1}: Train Acc={train_metrics['accuracy']:.4f}, Val Acc={val_results['accuracy']:.4f}")
    
    # Load best model and evaluate on test sets with plots
    model.load_state_dict(best_model_state)
    val_results = evaluate_model(model, val_loader, criterion, device, 
                                create_plots=True, plot_title=f"Adversarial-Validation-{train_dataset.dataset_name}")
    
    test_results = []
    for test_loader, test_name in zip(test_loaders, test_names):
        test_result = evaluate_model(model, test_loader, criterion, device, 
                                   create_plots=True, plot_title=f"Adversarial-CrossCorpus-{test_name}")
        test_results.append({
            'dataset': test_name,
            'results': test_result
        })
        print(f"   {test_name}: Acc={test_result['accuracy']:.4f}, UAR={test_result['uar']:.4f}")
    
    # Log to wandb
    if wandb.run:
        log_dict = {
            'validation/accuracy': val_results['accuracy'],
            'validation/uar': val_results['uar'],
        }
        
        # Add validation plots
        if 'confusion_matrix' in val_results:
            log_dict['validation/confusion_matrix'] = val_results['confusion_matrix']
        if 'difficulty_plot' in val_results:
            log_dict['validation/difficulty_plot'] = val_results['difficulty_plot']
            if 'difficulty_analysis' in val_results:
                analysis = val_results['difficulty_analysis']
                log_dict['validation/difficulty_correlation'] = analysis['difficulty_accuracy_correlation']
        
        # Add test results and plots
        for test_result in test_results:
            dataset_name = test_result['dataset'].lower()
            results = test_result['results']
            log_dict[f'test_{dataset_name}/accuracy'] = results['accuracy']
            log_dict[f'test_{dataset_name}/uar'] = results['uar']
            
            # Add test plots
            if 'confusion_matrix' in results:
                log_dict[f'test_{dataset_name}/confusion_matrix'] = results['confusion_matrix']
            if 'difficulty_plot' in results:
                log_dict[f'test_{dataset_name}/difficulty_plot'] = results['difficulty_plot']
                if 'difficulty_analysis' in results:
                    analysis = results['difficulty_analysis']
                    log_dict[f'test_{dataset_name}/difficulty_correlation'] = analysis['difficulty_accuracy_correlation']
        
        wandb.log(log_dict)
    
    results = {
        'validation': val_results,
        'test_results': test_results,
        'best_val_accuracy': best_val_acc
    }
    
    return results


def load_config_from_yaml(yaml_path, experiment_id=None):
    """Load configuration from YAML file and create Config object"""
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Check if this is a multi-experiment config
    if 'experiments' in yaml_config:
        if experiment_id is None:
            # List available experiments
            print("📋 Available experiments:")
            for i, exp in enumerate(yaml_config['experiments']):
                print(f"   {i}: {exp.get('name', exp.get('id', f'experiment_{i}'))}")
            raise ValueError("Please specify --experiment <id> when using multi-experiment config")
        
        # Find the specified experiment
        if isinstance(experiment_id, int):
            if 0 <= experiment_id < len(yaml_config['experiments']):
                experiment_config = yaml_config['experiments'][experiment_id]
            else:
                raise ValueError(f"Experiment index {experiment_id} out of range")
        else:
            # Find by name or id
            experiment_config = None
            for exp in yaml_config['experiments']:
                if exp.get('id') == experiment_id or exp.get('name') == experiment_id:
                    experiment_config = exp
                    break
            if experiment_config is None:
                raise ValueError(f"Experiment '{experiment_id}' not found")
        
        # Use the specific experiment config
        yaml_config = experiment_config
        print(f"🧪 Running experiment: {yaml_config.get('name', yaml_config.get('id', experiment_id))}")
    
    config = Config()
    
    # Define type conversions for config parameters
    float_params = ['learning_rate', 'weight_decay', 'dropout', 'val_split', 'adversarial_alpha']
    int_params = ['batch_size', 'num_epochs', 'hidden_dim', 'num_classes']
    bool_params = ['use_adversarial_learning']
    string_params = ['wandb_project', 'experiment_name']
    
    # Update config with YAML values with proper type conversion
    for key, value in yaml_config.items():
        if hasattr(config, key):
            # Apply type conversion
            if key in float_params and value is not None:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    print(f"⚠️  Could not convert {key}={value} to float, using as-is")
            elif key in int_params and value is not None:
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    print(f"⚠️  Could not convert {key}={value} to int, using as-is")
            elif key in bool_params and value is not None:
                if isinstance(value, str):
                    value = value.lower() in ['true', '1', 'yes', 'on']
                else:
                    value = bool(value)
            elif key in string_params and value is not None:
                value = str(value)
            
            setattr(config, key, value)
        elif key not in ['id', 'name', 'description', 'category']:  # Skip metadata
            print(f"⚠️  Unknown config parameter: {key}")
    
    # Set experiment name from YAML if available
    if 'name' in yaml_config:
        config.experiment_name = yaml_config['name']
    
    return config


class SimpleEmotionDataset(Dataset):
    """Simple dataset class for emotion recognition"""
    
    def __init__(self, dataset_name, split="train", config=None, Train=False):
        self.dataset_name = dataset_name
        self.split = split
        
        # Load HuggingFace dataset
        if dataset_name == "IEMO":
            self.hf_dataset = load_dataset("cairocode/IEMO_Emotion2Vec", split=split, trust_remote_code=True)
        elif dataset_name == "MSPI":
            self.hf_dataset = load_dataset("cairocode/MSPI_Emotion2Vec", split=split, trust_remote_code=True)
        elif dataset_name == "MSPP":
            self.hf_dataset = load_dataset("cairocode/MSPP_Emotion2vec_filtered", split=split, trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Process data
        self.data = []
        
        for item in self.hf_dataset:
            features = torch.tensor(item["emotion2vec_features"][0]["feats"], dtype=torch.float32)
            if Train == True:
                # Get speaker ID and calculate session directly
                if self.dataset_name == "IEMO":
                    speaker_id = item["speaker_id"]
                    session = (speaker_id - 1) // 2 + 1
                elif self.dataset_name == "MSPI":
                    speaker_id = item["speakerID"]
                    session = (speaker_id - 947) // 2 + 1
                elif self.dataset_name == "MSPP":
                    speaker_id = item["SpkrID"]
                    session = (speaker_id - 1) // 500 + 1
                else:
                    try:
                        speaker_id = item["speaker_id"]
                    except:
                        speaker_id = item.get("speakerID", item.get("SpkrID", 1))
                    session = (speaker_id - 1) // 2 + 1
            else:
                speaker_id = -1
                session = -1
            
            label = item["label"]
            
            # Get VAD values for difficulty calculation
            valence = item.get('valence', item.get('EmoVal', None))
            arousal = item.get('arousal', item.get('EmoAct', None))
            domination = item.get('domination', item.get('consensus_dominance', item.get('EmoDom', None)))
            
            # Replace NaN or None with 3
            def fix_vad(value):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return 3
                return value

            valence = fix_vad(valence)
            arousal = fix_vad(arousal)
            domination = fix_vad(domination)
            
            # Calculate difficulty if config is provided
            item_with_vad = {
                'label': label,
                'valence': valence,
                'arousal': arousal,
                'domination': domination
            }
            difficulty = calculate_difficulty(item_with_vad, config.expected_vad, config.difficulty_method, dataset=dataset_name)
            
            self.data.append({
                "features": features,
                "label": label,
                "speaker_id": speaker_id,
                "session": session,
                "dataset": dataset_name,
                "difficulty": difficulty,
                "valence": valence,
                "arousal": arousal,
                "domination": domination
            })
        
        print(f"✅ Loaded {len(self.data)} samples from {dataset_name}")
        
        # Print class and difficulty distribution for analysis
        class_counts = defaultdict(int)
        difficulty_by_class = defaultdict(list)
        for item in self.data:
            class_counts[item['label']] += 1
            difficulty_by_class[item['label']].append(item['difficulty'])
        
        print(f"📊 {dataset_name} Class Distribution:")
        class_names = ["neutral", "happy", "sad", "anger"]
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            difficulties = difficulty_by_class[class_id]
            avg_diff = np.mean(difficulties) if difficulties else 0
            print(f"   {class_names[class_id]}: {count} samples (avg difficulty: {avg_diff:.3f})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "features": item["features"],
            "label": torch.tensor(item["label"], dtype=torch.long),
            "speaker_id": item["speaker_id"],
            "session": item["session"],
            "dataset": item["dataset"],
            "difficulty": item["difficulty"]
        }


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment(config):
    """Run a single experiment with given config"""
    # Set seed for reproducibility
    seed = getattr(config, 'seed', 42)
    set_seed(seed)
    print(f"🔢 Random seed set to: {seed}")
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        name=config.experiment_name,
        config=vars(config)
    )
    
    print(f"🚀 Starting adversarial experiment: {config.experiment_name}")
    
    # Load datasets
    if config.train_dataset == "MSPI":
        train_dataset = SimpleEmotionDataset("MSPI", config=config, Train=True)
        test_datasets = [
            SimpleEmotionDataset("IEMO", config=config),
            SimpleEmotionDataset("MSPP", config=config)
        ]
        print(f"🚀 Adversarial Training: MSPI -> [IEMO, MSPP]")
    elif config.train_dataset == "IEMO":
        train_dataset = SimpleEmotionDataset("IEMO", config=config, Train=True)
        test_datasets = [
            SimpleEmotionDataset("MSPI", config=config),
            SimpleEmotionDataset("MSPP", config=config)
        ]
        print(f"🚀 Adversarial Training: IEMO -> [MSPI, MSPP]")
    elif config.train_dataset == "MSPP":
        train_dataset = SimpleEmotionDataset("MSPP", config=config, Train=True)
        test_datasets = [
            SimpleEmotionDataset("IEMO", config=config),
            SimpleEmotionDataset("MSPI", config=config)
        ]
        print(f"🚀 Adversarial Training: MSPP -> [IEMO, MSPI]")
    else:
        raise ValueError(f"Unknown train dataset: {config.train_dataset}")
    
    # Run adversarial cross-corpus evaluation
    results = run_adversarial_cross_corpus_evaluation(config, train_dataset, test_datasets)
    
    # Print final results
    print(f"\n{'='*60}")
    print(f"FINAL ADVERSARIAL RESULTS")
    print(f"{'='*60}")
    
    print(f"Validation Accuracy: {results['validation']['accuracy']:.4f}")
    print(f"Validation UAR: {results['validation']['uar']:.4f}")
    
    for test_result in results['test_results']:
        dataset_name = test_result['dataset']
        acc = test_result['results']['accuracy']
        uar = test_result['results']['uar']
        print(f"{dataset_name} Test Accuracy: {acc:.4f}")
        print(f"{dataset_name} Test UAR: {uar:.4f}")
    
    # Log final metrics
    final_log = {
        'final/validation_acc': results['validation']['accuracy'],
        'final/validation_uar': results['validation']['uar']
    }
    for test_result in results['test_results']:
        dataset_name = test_result['dataset'].lower()
        final_log[f'final/{dataset_name}_acc'] = test_result['results']['accuracy']
        final_log[f'final/{dataset_name}_uar'] = test_result['results']['uar']
    wandb.log(final_log)
    
    wandb.finish()
    return results


def main(config_path=None, experiment_id=None):
    """Main training function"""
    if config_path:
        config = load_config_from_yaml(config_path, experiment_id)
        return run_experiment(config)
    else:
        print("📄 Using default config")
        config = Config()
        return run_experiment(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Emotion Recognition Learning")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML configuration file")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment ID/name for multi-experiment configs")
    
    args = parser.parse_args()
    
    try:
        # Convert experiment to int if it's a number
        experiment_id = args.experiment
        if experiment_id and experiment_id.isdigit():
            experiment_id = int(experiment_id)
        
        main(args.config, experiment_id)
    except Exception as e:
        import traceback
        print(f"💥 Fatal error: {e}")
        print(f"🔍 Full traceback:")
        print(traceback.format_exc())
        sys.exit(1)