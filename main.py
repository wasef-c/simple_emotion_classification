#!/usr/bin/env python3
"""
Simple emotion recognition training script
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

# Add parent directory to path to access original dataset classes
sys.path.append(str(Path(__file__).parent.parent))

from datasets import load_dataset
from collections import defaultdict
import wandb

# Import local modules
from config import Config
from model import * #SimpleEmotionClassifier
from functions import * #(evaluate_model, get_session_splits, calculate_metrics, calculate_difficulty, 
                    #   get_curriculum_pacing_function, create_curriculum_subset, create_confusion_matrix, create_difficulty_accuracy_plot)


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
    float_params = ['learning_rate', 'weight_decay', 'dropout', 'val_split']
    int_params = ['batch_size', 'num_epochs', 'hidden_dim', 'num_classes', 'curriculum_epochs']
    bool_params = ['use_curriculum_learning', 'use_difficulty_scaling', 'use_speaker_disentanglement']
    string_params = ['wandb_project', 'experiment_name', 'curriculum_type', 'difficulty_method', 'curriculum_pacing']
    
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


def run_all_experiments_from_yaml(yaml_path):
    """Run all experiments from a multi-experiment YAML file"""
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    if 'experiments' not in yaml_config:
        raise ValueError("YAML file doesn't contain multiple experiments")
    
    print(f"🚀 Running {len(yaml_config['experiments'])} experiments from {yaml_path}")
    results = []
    
    for i, experiment in enumerate(yaml_config['experiments']):
        exp_name = experiment.get('name', experiment.get('id', f'experiment_{i}'))
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT {i+1}/{len(yaml_config['experiments'])}: {exp_name}")
        print(f"{'='*60}")
        
        try:
            config = load_config_from_yaml(yaml_path, i)
            result = run_experiment(config)
            results.append({
                'experiment_id': i,
                'name': exp_name,
                'result': result,
                'status': 'completed'
            })
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            print(f"❌ Experiment {exp_name} failed: {e}")
            print(f"🔍 Full traceback:")
            print(full_traceback)
            results.append({
                'experiment_id': i,
                'name': exp_name,
                'result': None,
                'status': 'failed',
                'error': str(e),
                'traceback': full_traceback
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = len(results) - completed
    print(f"✅ Completed: {completed}")
    print(f"❌ Failed: {failed}")
    
    return results


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
    
    print(f"🚀 Starting experiment: {config.experiment_name}")
    print(f"📊 Evaluation Mode: {config.evaluation_mode.upper()}")
    
    # Load datasets
    if config.train_dataset == "MSPI":
        train_dataset = SimpleEmotionDataset("MSPI", config=config, Train=True)
        # For cross-corpus, we want to test on both other datasets
        if config.evaluation_mode == "cross_corpus":
            test_datasets = [
                SimpleEmotionDataset("IEMO", config=config),
                SimpleEmotionDataset("MSPP", config=config)
            ]
            print(f"🚀 Training: MSPI -> [IEMO, MSPP]")
        else:
            test_dataset = SimpleEmotionDataset("IEMO", config=config)
            print(f"🚀 Training: MSPI -> IEMO")
    elif config.train_dataset == "IEMO":
        train_dataset = SimpleEmotionDataset("IEMO", config=config, Train=True)
        if config.evaluation_mode == "cross_corpus":
            test_datasets = [
                SimpleEmotionDataset("MSPI", config=config),
                SimpleEmotionDataset("MSPP", config=config)
            ]
            print(f"🚀 Training: IEMO -> [MSPI, MSPP]")
        else:
            test_dataset = SimpleEmotionDataset("MSPI", config=config)
            print(f"🚀 Training: IEMO -> MSPI")
    elif config.train_dataset == "MSPP":
        train_dataset = train_dataset("MSPP", config=config, Train=True)
        if config.evaluation_mode == "cross_corpus":
            test_datasets = [
                SimpleEmotionDataset("IEMO", config=config),
                SimpleEmotionDataset("MSPI", config=config)
            ]
            print(f"🚀 Training: MSPP -> [IEMO, MSPI]")
        else:
            test_dataset = SimpleEmotionDataset("IEMO", config=config)
            print(f"🚀 Training: MSPP -> IEMO")
    else:
        raise ValueError(f"Unknown train dataset: {config.train_dataset}")
    
    # Run evaluation based on mode
    if config.evaluation_mode == "loso":
        results = run_loso_evaluation(config, train_dataset, test_dataset)
    elif config.evaluation_mode == "cross_corpus":
        results = run_cross_corpus_evaluation(config, train_dataset, test_datasets)
    elif config.evaluation_mode == "both":
        print("\n" + "="*60)
        print("RUNNING LOSO EVALUATION")
        print("="*60)
        loso_results = run_loso_evaluation(config, train_dataset, test_dataset)
        
        print("\n" + "="*60) 
        print("RUNNING CROSS-CORPUS EVALUATION")
        print("="*60)
        cross_corpus_results = run_cross_corpus_evaluation(config, train_dataset, [test_dataset])
        
        results = {
            'loso': loso_results,
            'cross_corpus': cross_corpus_results
        }
    else:
        raise ValueError(f"Unknown evaluation mode: {config.evaluation_mode}")
    
    # Print final results based on evaluation mode
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - {config.evaluation_mode.upper()}")
    print(f"{'='*60}")
    
    if config.evaluation_mode == "loso":
        print(f"LOSO Accuracy: {results['loso_accuracy_mean']:.4f} ± {results['loso_accuracy_std']:.4f}")
        print(f"LOSO UAR: {results['loso_uar_mean']:.4f} ± {results['loso_uar_std']:.4f}")
        
        # Log final metrics
        wandb.log({
            'final/loso_acc_mean': results['loso_accuracy_mean'],
            'final/loso_acc_std': results['loso_accuracy_std'],
            'final/loso_uar_mean': results['loso_uar_mean'],
            'final/loso_uar_std': results['loso_uar_std'],
        })
        
    elif config.evaluation_mode == "cross_corpus":
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
        
    elif config.evaluation_mode == "both":
        print("LOSO Results:")
        loso = results['loso']
        print(f"  LOSO Accuracy: {loso['loso_accuracy_mean']:.4f} ± {loso['loso_accuracy_std']:.4f}")
        print(f"  LOSO UAR: {loso['loso_uar_mean']:.4f} ± {loso['loso_uar_std']:.4f}")
        
        print("\nCross-Corpus Only Results:")
        cross = results['cross_corpus']
        print(f"  Validation Accuracy: {cross['validation']['accuracy']:.4f}")
        print(f"  Validation UAR: {cross['validation']['uar']:.4f}")
        for test_result in cross['test_results']:
            dataset_name = test_result['dataset']
            acc = test_result['results']['accuracy']
            uar = test_result['results']['uar']
            print(f"  {dataset_name} Test Accuracy: {acc:.4f}")
            print(f"  {dataset_name} Test UAR: {uar:.4f}")
        
        # Log both sets of metrics
        final_log = {
            'final/loso_acc_mean': loso['loso_accuracy_mean'],
            'final/loso_uar_mean': loso['loso_uar_mean'],
            'final/validation_acc': cross['validation']['accuracy'],
            'final/validation_uar': cross['validation']['uar']
        }
        for test_result in cross['test_results']:
            dataset_name = test_result['dataset'].lower()
            final_log[f'final/{dataset_name}_cross_only_acc'] = test_result['results']['accuracy']
            final_log[f'final/{dataset_name}_cross_only_uar'] = test_result['results']['uar']
        wandb.log(final_log)
    
    wandb.finish()
    return results


class SimpleEmotionDataset(Dataset):
    """Simple dataset class for emotion recognition"""
    
    def __init__(self, dataset_name, split="train", config=None, Train = False):
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
                    # Fallback for other datasets
                    try:
                        speaker_id = item["speaker_id"]
                    except:
                        speaker_id = item.get("speakerID", item.get("SpkrID", 1))
                    session = (speaker_id - 1) // 2 + 1
            else:
                speaker_id = -1  # Use -1 instead of None for test datasets
                session = -1     # Use -1 instead of None for test datasets
            
            label = item["label"]
            
            # Get curriculum order from dataset
            curriculum_order = item.get('curriculum_order', 0.5)  # Default to middle if missing
            
            # Get VAD values for difficulty calculation
            valence = item.get('valence', item.get('EmoVal', None))
            arousal = item.get('arousal',  item.get('EmoAct', None))
            domination = item.get('domination', item.get('consensus_dominance', item.get('EmoDom', None)))

            
            # Replace NaN or None with 3
            def fix_vad(value):
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    return 3
                return value

            valence = fix_vad(valence)
            arousal = fix_vad(arousal)
            domination = fix_vad(domination)
            item_with_vad = {
                'label': label,
                'valence': valence,
                'arousal': arousal,
                'domination': domination
            }
            difficulty = calculate_difficulty(item_with_vad, config.expected_vad, config.difficulty_method, dataset = dataset_name)
            
            self.data.append({
                "features": features,
                "label": label,
                "speaker_id": speaker_id,
                "session": session,
                "dataset": dataset_name,
                "difficulty": difficulty,
                "curriculum_order": curriculum_order,
                "valence": valence,
                "arousal": arousal,
                "domination": domination
            })
        
        print(f"✅ Loaded {len(self.data)} samples from {dataset_name}")
        
        # Print session distribution for debugging
        session_counts = defaultdict(int)
        for item in self.data:
            session_counts[item['session']] += 1
        
        print(f"📊 {dataset_name} Sessions:")
        for session_id in sorted(session_counts.keys()):
            print(f"   Session {session_id}: {session_counts[session_id]} samples")
    
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
            "difficulty": item["difficulty"],
            "curriculum_order": item["curriculum_order"]
        }


def train_epoch(model, data_loader, criterion, optimizer, scheduler, device, use_difficulty_scaling = False):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    for batch in data_loader:
        # print(batch)
        features = batch['features'].to(device)
        batch_labels = batch['label'].to(device)
        difficulties = batch['difficulty'].to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        # print(f"LOGITSSSS -- {logits}")
        # loss = criterion(logits, batch_labels) # difficulties
        loss_per_sample = criterion(logits, batch_labels)  # reduction='none'
    
        # weighted loss
        if use_difficulty_scaling:
            loss = (loss_per_sample * (1+difficulties)).mean()
        else:
            loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']

        log_dict = {
            "loss": loss,
            "batch difficulty": difficulties.mean(),
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


def run_loso_evaluation(config, train_dataset, test_dataset):
    """Run Leave-One-Session-Out evaluation with curriculum learning"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print(f"📚 Curriculum Learning: {'Enabled' if config.use_curriculum_learning else 'Disabled'}")
    
    # Get session splits
    train_sessions = get_session_splits(train_dataset, train_dataset.dataset_name)
    
    session_results = []
    
    for test_session in sorted(train_sessions.keys()):
        print(f"\n📊 LOSO Session {test_session} ({train_dataset.dataset_name})")
        
        # Create train/test splits
        train_indices = []
        for session_id, indices in train_sessions.items():
            if session_id != test_session:
                train_indices.extend(indices)
        
        test_indices = train_sessions[test_session]
        
        # Get difficulties for curriculum learning (use the difficulty we calculated in dataset)
        train_difficulties = [train_dataset.data[i]['difficulty'] for i in train_indices]
        
        # Create base datasets
        train_subset = Subset(train_dataset, train_indices)
        test_subset = Subset(train_dataset, test_indices)
        cross_corpus_loader = create_data_loader(test_dataset, config.batch_size, shuffle=False, 
                                                 use_speaker_disentanglement=False)  # No speaker grouping for test
        loso_loader = create_data_loader(test_subset, config.batch_size, shuffle=False, 
                                       use_speaker_disentanglement=False)  # No speaker grouping for test
        
        # Get actual feature dimension from dataset
        sample_features = train_dataset[0]['features']
        if len(sample_features.shape) == 2:
            # If 2D, use the last dimension (sequence_length, feature_dim)
            actual_input_dim = sample_features.shape[-1]
        else:
            # If 1D, use the full dimension
            actual_input_dim = sample_features.shape[0]
        
        print(f"🔍 Using input_dim: {actual_input_dim} (detected from {train_dataset.dataset_name})")
        
        # Initialize model
        model = SimpleEmotionClassifier(
            input_dim=actual_input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        ).to(device)
        
        criterion = FocalLossAutoWeights(num_classes=4, gamma=2.0, reduction='none', device=device)

        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        # Get curriculum pacing function
        pacing_function = get_curriculum_pacing_function(config.curriculum_pacing)
        
        # Training loop with curriculum learning
        best_loso_acc = 0
        for epoch in range(config.num_epochs):
            
            # Create curriculum subset if enabled
            if config.use_curriculum_learning and epoch < config.curriculum_epochs:
                curriculum_indices = create_curriculum_subset(
                    train_indices, train_difficulties, epoch, 
                    config.curriculum_epochs, pacing_function
                )
                curriculum_train_indices = [train_indices[i] for i in curriculum_indices]
                curriculum_subset = Subset(train_dataset, curriculum_train_indices)
                train_loader = create_data_loader(curriculum_subset, config.batch_size, shuffle=True,
                                                use_speaker_disentanglement=config.use_speaker_disentanglement)
                
                fraction = pacing_function(epoch, config.curriculum_epochs)
                print(f"   Epoch {epoch+1}: Using {len(curriculum_indices)}/{len(train_indices)} samples ({fraction:.2f})")
            else:
                # Use all training data
                train_loader = create_data_loader(train_subset, config.batch_size, shuffle=True,
                                                use_speaker_disentanglement=config.use_speaker_disentanglement)
                if epoch == config.curriculum_epochs:
                    print(f"   Epoch {epoch+1}: Curriculum complete, using all training data")
            
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, use_difficulty_scaling= config.use_difficulty_scaling)
            loso_results = evaluate_model(model, loso_loader, criterion, device, create_plots=False)
            
            if loso_results['uar'] > best_loso_acc:
                best_loso_acc = loso_results['accuracy']
                best_model_state = model.state_dict().copy()
        
        # Load best model and evaluate with plots
        model.load_state_dict(best_model_state)
        loso_results = evaluate_model(model, loso_loader, criterion, device, 
                                    create_plots=True, plot_title=f"LOSO-Session-{test_session}-{train_dataset.dataset_name}")
        
        print(f"   LOSO: Acc={loso_results['accuracy']:.4f}, UAR={loso_results['uar']:.4f}")
        
        session_results.append({
            'session': test_session,
            'loso': loso_results,
        })
        
        # Log to wandb
        if wandb.run:
            log_dict = {
                f"session_{test_session}/loso_acc": loso_results['accuracy'],
                f"session_{test_session}/loso_uar": loso_results['uar'],
            }
            
            # Add plots if available
            if 'confusion_matrix' in loso_results:
                log_dict[f"session_{test_session}/confusion_matrix"] = loso_results['confusion_matrix']
            
            if 'difficulty_plot' in loso_results:
                log_dict[f"session_{test_session}/difficulty_plot"] = loso_results['difficulty_plot']
                
                # Add difficulty analysis metrics
                if 'difficulty_analysis' in loso_results:
                    analysis = loso_results['difficulty_analysis']
                    log_dict[f"session_{test_session}/difficulty_correlation"] = analysis['difficulty_accuracy_correlation']
                    log_dict[f"session_{test_session}/correlation_p_value"] = analysis['correlation_p_value']
            
            wandb.log(log_dict)
    
    # Calculate averages
    loso_accs = [r['loso']['accuracy'] for r in session_results]
    loso_uars = [r['loso']['uar'] for r in session_results]

    
    results = {
        'loso_accuracy_mean': np.mean(loso_accs),
        'loso_accuracy_std': np.std(loso_accs),
        'loso_uar_mean': np.mean(loso_uars),
        'loso_uar_std': np.std(loso_uars),
        'session_results': session_results
    }
    
    return results


def run_cross_corpus_evaluation(config, train_dataset, test_datasets):
    """Run cross-corpus evaluation with train/val split on training dataset"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    print(f"📚 Curriculum Learning: {'Enabled' if config.use_curriculum_learning else 'Disabled'}")
    print(f"📊 Cross-Corpus Only Mode: Train={1-config.val_split:.0%}, Val={config.val_split:.0%}")
    
    # Create train/validation split
    total_samples = len(train_dataset)
    indices = list(range(total_samples))
 
    np.random.shuffle(indices)
    
    val_size = int(total_samples * config.val_split)
    train_indices = indices[val_size:]
    if config.curriculum_type == "preset_order":
    # Sort indices by curriculum_order
        train_indices.sort(key=lambda i: train_dataset[i]["curriculum_order"])
    val_indices = indices[:val_size]
    
    print(f"📈 Training samples: {len(train_indices)}")
    print(f"📋 Validation samples: {len(val_indices)}")
    
    # Get difficulties for curriculum learning (use the difficulty we calculated in dataset)  
    train_difficulties = [train_dataset.data[i]['difficulty'] for i in train_indices]
    
    # Create datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)
    
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
        # If 2D, use the last dimension (sequence_length, feature_dim)
        actual_input_dim = sample_features.shape[-1]
    else:
        # If 1D, use the full dimension
        actual_input_dim = sample_features.shape[0]
    
    print(f"🔍 Using input_dim: {actual_input_dim} (detected from {train_dataset.dataset_name})")
    
    # Initialize model
    model = SimpleEmotionClassifier(
        input_dim=actual_input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout
    ).to(device)
    
    # Calculate class weights = (1/frequency) * average_difficulty
    class_counts = [0, 0, 0, 0]
    class_difficulties = [[], [], [], []]
    
    for item in train_dataset.data:
        label = item['label']
        difficulty = item['difficulty']
        class_counts[label] += 1
        class_difficulties[label].append(difficulty)
    
    class_weights = []
    for i in range(4):
        # freq_weight = 1.0 / class_counts[i] if class_counts[i] > 0 else 1.0
        freq_ratio = class_counts[i] / total_samples
        # proportion of this class
        freq_weight = (1.0 / freq_ratio) / 4
        avg_difficulty = sum(class_difficulties[i]) / len(class_difficulties[i]) if class_difficulties[i] else 1.0
        class_weights.append((1+freq_weight) * avg_difficulty)
        print(f"📊 freq_weight: {freq_weight}")
    
    # Normalize weights
    total_weight = sum(class_weights)
    class_weights = [w / total_weight * 4 for w in class_weights]  # Scale to average of 1.0
    
    class_weights = torch.tensor(class_weights).to(device)
    print(f"📊 Class weights (freq × difficulty): {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    
    # Get curriculum pacing function
    pacing_function = get_curriculum_pacing_function(config.curriculum_pacing)
    
    # Training loop with curriculum learning
    best_val_acc = 0
    for epoch in range(config.num_epochs):
        
        # Create curriculum subset if enabled
        if config.use_curriculum_learning and epoch < config.curriculum_epochs:
            if config.curriculum_type == "preset_order":
                use_preset = True
            else:
                use_preset = False
            curriculum_indices = create_curriculum_subset(
                train_indices, train_difficulties, epoch, 
                config.curriculum_epochs, pacing_function, use_preset
            )
            curriculum_train_indices = [train_indices[i] for i in curriculum_indices]
            curriculum_subset = Subset(train_dataset, curriculum_train_indices)
            train_loader = DataLoader(curriculum_subset, batch_size=config.batch_size, shuffle=False)
            
            fraction = pacing_function(epoch, config.curriculum_epochs)
            print(f"   Epoch {epoch+1}: Using {len(curriculum_indices)}/{len(train_indices)} samples ({fraction:.2f})")
        else:
            # Use all training data
            train_loader = create_data_loader(train_subset, config.batch_size, shuffle=True,
                                             use_speaker_disentanglement=config.use_speaker_disentanglement)
            if epoch == config.curriculum_epochs:
                print(f"   Epoch {epoch+1}: Curriculum complete, using all training data")
        
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer,scheduler, device, use_difficulty_scaling = config.use_difficulty_scaling)
        
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
                                create_plots=True, plot_title=f"Validation-{train_dataset.dataset_name}")
    
    test_results = []
    for test_loader, test_name in zip(test_loaders, test_names):
        test_result = evaluate_model(model, test_loader, criterion, device, 
                                   create_plots=True, plot_title=f"CrossCorpus-{test_name}")
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
            prefix = f"{train_dataset.dataset_name}_TO_{dataset_name}"
            results = test_result['results']
            log_dict[f'{prefix}/accuracy'] = results['accuracy']
            log_dict[f'{prefix}/uar'] = results['uar']
            
            # Add test plots
            if 'confusion_matrix' in results:
                log_dict[f'{prefix}/confusion_matrix'] = results['confusion_matrix']
            if 'difficulty_plot' in results:
                log_dict[f'{prefix}/difficulty_plot'] = results['difficulty_plot']
                if 'difficulty_analysis' in results:
                    analysis = results['difficulty_analysis']
                    log_dict[f'{prefix}/difficulty_correlation'] = analysis['difficulty_accuracy_correlation']
        
        wandb.log(log_dict)
    
    results = {
        'validation': val_results,
        'test_results': test_results,
        'best_val_accuracy': best_val_acc
    }
    
    return results


def main(config_path=None, experiment_id=None, all_experiments=False):
    """Main training function"""
    if config_path:
        if all_experiments:
            return run_all_experiments_from_yaml(config_path)
        else:
            config = load_config_from_yaml(config_path, experiment_id)
            return run_experiment(config)
    else:
        print("📄 Using default config")
        config = Config()
        return run_experiment(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Emotion Recognition with Curriculum Learning")
    parser.add_argument("--config", "-c", type=str, help="Path to YAML configuration file")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment ID/name for multi-experiment configs")
    parser.add_argument("--all", "-a", action="store_true", help="Run all experiments in config file")
    
    args = parser.parse_args()
    
    try:
        # Convert experiment to int if it's a number
        experiment_id = args.experiment
        if experiment_id and experiment_id.isdigit():
            experiment_id = int(experiment_id)
        
        main(args.config, experiment_id, args.all)
    except Exception as e:
        import traceback
        print(f"💥 Fatal error: {e}")
        print(f"🔍 Full traceback:")
        print(traceback.format_exc())
        sys.exit(1)