#!/usr/bin/env python3
"""
Utility functions for emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import random
from torch.utils.data import DataLoader, Dataset, Subset


def calculate_metrics(predictions, labels):
    """Calculate basic classification metrics"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    accuracy = accuracy_score(labels, predictions)
    # UAR = macro average F1 (unweighted average recall)
    uar = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'uar': uar, 
        'f1_weighted': f1_weighted
    }


def evaluate_model(model, data_loader, criterion, device, return_difficulties=True, create_plots=True, plot_title=""):
    """Evaluate model on a dataset with optional confusion matrix and difficulty plots"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    difficulties = []
    
    with torch.no_grad():
        for batch in data_loader:
            features = batch['features'].to(device)
            batch_labels = batch['label'].to(device)
            
            logits = model(features)
            loss = criterion(logits, batch_labels)
            
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
            labels.extend(batch_labels.cpu().numpy())
            
            # Collect difficulties if requested
            if return_difficulties:
                batch_difficulties = batch.get('difficulty', [0.5] * len(batch_labels))
                if torch.is_tensor(batch_difficulties):
                    batch_difficulties = batch_difficulties.cpu().numpy()
                difficulties.extend(batch_difficulties)
    
    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(predictions, labels)
    
    results = {
        'loss': avg_loss,
        'predictions': predictions,
        'labels': labels,
        'difficulties': difficulties if return_difficulties else None,
        **metrics
    }
    
    # Create plots if requested
    if create_plots and plot_title:
        # Confusion matrix
        confusion_matrix_plot = create_confusion_matrix(predictions, labels, plot_title)
        results['confusion_matrix'] = confusion_matrix_plot
        
        # Difficulty vs accuracy plot (only if we have difficulties)
        if return_difficulties and len(difficulties) > 0:
            difficulty_plot, difficulty_analysis = create_difficulty_accuracy_plot(
                predictions, labels, difficulties, plot_title
            )
            results['difficulty_plot'] = difficulty_plot
            results['difficulty_analysis'] = difficulty_analysis
    
    return results


# def calculate_difficulty(item, expected_vad, method="euclidean_distance", dataset=None):
#     """Calculate sample difficulty based on VAD values"""

#     label = item.get('label', 0)
#     valence = item.get('valence', item.get('EmoVal', None))
#     arousal = item.get('arousal',  item.get('EmoAct', None))
#     domination = item.get('domination', item.get('consensus_domination', item.get('EmoDom', None)))
#     if dataset == "MSPP":
#         valence = valence*5/7
#         arousal = arousal*5/7
#         domination = domination*5/7

#     # If VAD values are missing, return neutral difficulty
#     if any(v is None for v in [valence, arousal, domination]):
#         print([valence, arousal, domination])
    
#     actual_vad = [valence, arousal, domination]
#     expected = expected_vad.get(label, None)
#     if expected == None:
#         print("ERROR: expected = ", expected)
    
#     # if method == "euclidean_distance":
#     # Calculate Euclidean distance from expected VAD
#     distance = math.sqrt(sum((a - e) ** 2 for a, e in zip(actual_vad, expected)))
        
#     # Normalize to 0-1 range (assuming max distance is about 3.0)
    
#     difficulty = distance
    
#     return difficulty

import math

def calculate_difficulty(item, expected_vad, method="euclidean_distance", dataset=None):
    """Calculate sample difficulty based on VAD values"""
    
    label = item.get('label', 0)
    valence = item.get('valence', item.get('EmoVal'))
    arousal = item.get('arousal', item.get('EmoAct'))
    domination = item.get('domination', item.get('consensus_domination', item.get('EmoDom')))

    if dataset == "MSPP":
        try:
            valence = (valence - 1) * 4 / 6 + 1
            arousal = (arousal - 1) * 4 / 6 + 1
            domination = (domination - 1) * 4 / 6 + 1
        except TypeError:
            print(f"Scaling failed: {valence}, {arousal}, {domination}")

    actual_vad = [valence, arousal, domination]

    # Check for missing values
    if any(v is None or not isinstance(v, (int, float)) for v in actual_vad):
        print(f"Missing or invalid VAD values for label {label}: {actual_vad}")
        return 0.0  # neutral difficulty

    expected = expected_vad.get(label)
    if expected is None or any(e is None for e in expected):
        print(f"Missing expected VAD for label {label}")
        return 0.0

    # Euclidean distance
    distance = math.sqrt(sum((float(a) - float(e)) ** 2 for a, e in zip(actual_vad, expected)))

    # Guard against NaN or inf
    if math.isnan(distance) or math.isinf(distance):
        print(f"Invalid distance for label {label}: {distance} -- {actual_vad} -- {expected}")
        return 0.0

    return distance

def get_curriculum_pacing_function(pacing_type):
    """Get pacing function for curriculum learning"""
    if pacing_type == "linear":
        return lambda epoch, total_epochs: min(2.0, (epoch + 1) / total_epochs)
    elif pacing_type == "sqrt":
        return lambda epoch, total_epochs: min(2.0, math.sqrt((epoch + 1) / total_epochs))
    elif pacing_type == "log":
        return lambda epoch, total_epochs: min(2.0, math.log(epoch + 2) / math.log(total_epochs + 1))
    else:
        return lambda epoch, total_epochs: 1.0  # No pacing


def create_curriculum_subset(dataset, difficulties, epoch, total_curriculum_epochs, pacing_function):
    """Create subset of data based on curriculum learning strategy"""
    if epoch >= total_curriculum_epochs:
        # After curriculum epochs, use all data
        return list(range(len(dataset)))
    
    # Calculate the fraction of data to include
    fraction = pacing_function(epoch, total_curriculum_epochs)
    log_dict = {
        "curriculum_fraction": fraction
    }
    wandb.log(log_dict)
    num_samples = int(len(dataset) * fraction)
    # Step 1: shuffle all indices
    shuffled_indices = list(range(len(difficulties)))
    random.shuffle(shuffled_indices)
    
    # Step 2: sort the shuffled indices by difficulty (keeps random tie-breaks)
    sorted_indices = sorted(shuffled_indices, key=lambda i: difficulties[i])
    
    # Step 3: take the easiest subset
    return sorted_indices[:num_samples]


def create_confusion_matrix(predictions, labels, title, class_names=None):
    """Create confusion matrix plot"""
    if class_names is None:
        class_names = ["Neutral", "Happy", "Sad", "Anger"]
    
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2, 3])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix - {title}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    
    # Create wandb image
    wandb_image = wandb.Image(plt, caption=f"Confusion Matrix - {title}")
    plt.close()
    
    return wandb_image


def create_difficulty_accuracy_plot(predictions, labels, difficulties, title, num_buckets=20):
    """Create difficulty vs accuracy plot with bucketed analysis"""
    predictions = np.array(predictions)
    labels = np.array(labels)
    difficulties = np.array(difficulties)
    
    # Create equal-size buckets based on difficulty quantiles
    difficulty_bins = np.percentile(difficulties, np.linspace(0, 100, num_buckets + 1))
    bucket_accuracies = []
    bucket_centers = []
    bucket_sizes = []
    
    for i in range(num_buckets):
        # Find samples in this difficulty bucket
        if i == num_buckets - 1:
            # Last bucket includes the maximum
            mask = (difficulties >= difficulty_bins[i]) & (difficulties <= difficulty_bins[i + 1])
        else:
            mask = (difficulties >= difficulty_bins[i]) & (difficulties < difficulty_bins[i + 1])
        
        if mask.sum() > 0:
            bucket_preds = predictions[mask]
            bucket_labels = labels[mask]
            bucket_acc = accuracy_score(bucket_labels, bucket_preds)
            bucket_center = (difficulty_bins[i] + difficulty_bins[i + 1]) / 2
            
            bucket_accuracies.append(bucket_acc)
            bucket_centers.append(bucket_center)
            bucket_sizes.append(mask.sum())
        else:
            bucket_accuracies.append(0.0)
            bucket_centers.append((difficulty_bins[i] + difficulty_bins[i + 1]) / 2)
            bucket_sizes.append(0)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bucket_centers, bucket_accuracies, width=0.8/num_buckets, alpha=0.7, color='skyblue')
    
    # Add sample counts on top of bars
    for i, (bar, size) in enumerate(zip(bars, bucket_sizes)):
        if size > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'n={size}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Difficulty Score')
    plt.ylabel('Accuracy')
    plt.title(f'Difficulty vs Accuracy - {title}')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Add overall accuracy line
    overall_acc = accuracy_score(labels, predictions)
    plt.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'Overall Accuracy: {overall_acc:.3f}')
    
    # Add line of best fit
    valid_buckets = [(center, acc) for center, acc, size in zip(bucket_centers, bucket_accuracies, bucket_sizes) if size > 0]
   
    
    # Calculate correlation between difficulty and accuracy
    from scipy.stats import pearsonr
    valid_buckets = [(center, acc) for center, acc, size in zip(bucket_centers, bucket_accuracies, bucket_sizes) if size > 0]
    if len(valid_buckets) > 2:
        centers, accs = zip(*valid_buckets)
        correlation, p_value = pearsonr(centers, accs)
    else:
        correlation, p_value = 0.0, 1.0
    
    if len(valid_buckets) > 1:
        centers, accs = zip(*valid_buckets)
        z = np.polyfit(centers, accs, 1)  # Linear fit
        p = np.poly1d(z)
        x_line = np.linspace(min(centers), max(centers), 100)
        plt.plot(x_line, p(x_line), color='orange', linestyle='-', alpha=0.8, linewidth=2,
                label=f'Best Fit (r={correlation:.3f})')
    
    plt.legend()
    
    # Create wandb image
    wandb_image = wandb.Image(plt, caption=f"Difficulty vs Accuracy - {title}")
    plt.close()
    
    analysis_data = {
        'difficulty_accuracy_correlation': correlation,
        'correlation_p_value': p_value,
        'bucket_accuracies': bucket_accuracies,
        'bucket_centers': bucket_centers,
        'bucket_sizes': bucket_sizes,
        'overall_accuracy': overall_acc
    }
    
    return wandb_image, analysis_data


def get_session_splits(dataset, dataset_name):
    """Get session-based splits for LOSO evaluation"""
    session_splits = defaultdict(list)
    
    for i, item in enumerate(dataset.data):
        session = item['session']
        if session is not None:
            session_splits[session].append(i)
    
    print(f"📊 {dataset_name} Sessions: {sorted(session_splits.keys())}")
    for session_id in sorted(session_splits.keys()):
        print(f"   Session {session_id}: {len(session_splits[session_id])} samples")
    
    return dict(session_splits)



class FocalLossAutoWeights(nn.Module):
    def __init__(self, num_classes, gamma=2.0, reduction='none', device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction
        self.device = device

    def forward(self, logits, targets):
        """
        logits: [batch_size, num_classes]
        targets: [batch_size] (class indices)
        """
        batch_size = targets.size(0)

        # Calculate class frequencies in the batch
        with torch.no_grad():
            counts = torch.bincount(targets, minlength=self.num_classes).float()
            # Avoid division by zero
            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
            class_weights = 1.0 / counts  # inverse frequency
            class_weights = class_weights / class_weights.sum()  # normalize weights to sum to 1
            # class_weights = 1- class_weights + 0.1
            class_weights[1] = class_weights[1] *3
            class_weights[2] = class_weights[2] *3


            class_weights = class_weights.to(self.device)
            

        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = torch.exp(log_probs)  # [B, C]

        targets = targets.long()
        log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # [B]
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # [B]

        focal_term = (1 - pt) ** self.gamma  # [B]

        # Get weights for each target in the batch
        weights = class_weights[targets]  # [B]

        loss = weights * focal_term * log_pt  # [B]
        loss = focal_term * log_pt  # [B]


        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SpeakerGroupedSampler:
    """Sampler that groups samples by speaker_id for speaker disentanglement"""
    
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by speaker_id
        self.speaker_groups = defaultdict(list)
        for idx, item in enumerate(dataset):
            speaker_id = item['speaker_id'] if isinstance(item, dict) else dataset[idx]['speaker_id']
            # Only group valid speaker IDs (not -1 for test datasets)
            if speaker_id != -1:
                self.speaker_groups[speaker_id].append(idx)
        
        print(f"🗣️  Speaker Disentanglement: Found {len(self.speaker_groups)} speakers")
        for speaker_id, indices in self.speaker_groups.items():
            print(f"   Speaker {speaker_id}: {len(indices)} samples")
    
    def __iter__(self):
        """Generate batches grouped by speaker"""
        all_batches = []
        
        # Create batches for each speaker
        for speaker_id, indices in self.speaker_groups.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            
            # Create batches from this speaker's samples
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                all_batches.append(batch)
        
        # Shuffle the order of batches (not within batches)
        if self.shuffle:
            random.shuffle(all_batches)
        
        # Yield batches
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        """Total number of batches"""
        total_batches = 0
        for indices in self.speaker_groups.values():
            total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        return total_batches


class SpeakerGroupedDataLoader:
    """DataLoader wrapper that uses SpeakerGroupedSampler"""
    
    def __init__(self, dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = SpeakerGroupedSampler(dataset, batch_size, shuffle)
        self.num_workers = num_workers
    
    def __iter__(self):
        """Iterate through speaker-grouped batches"""
        for batch_indices in self.sampler:
            batch = {
                'features': [],
                'label': [],
                'speaker_id': [],
                'session': [],
                'dataset': [],
                'difficulty': []
            }
            
            for idx in batch_indices:
                item = self.dataset[idx]
                
                batch['features'].append(item['features'])
                batch['label'].append(item['label'])
                batch['speaker_id'].append(item['speaker_id'])
                batch['session'].append(item['session'])
                batch['dataset'].append(item['dataset'])
                batch['difficulty'].append(item['difficulty'])
            
            # Stack tensors
            batch['features'] = torch.stack(batch['features'])
            batch['label'] = torch.stack(batch['label'])
            batch['speaker_id'] = torch.tensor(batch['speaker_id'])
            batch['session'] = torch.tensor(batch['session'])
            batch['difficulty'] = torch.tensor(batch['difficulty'], dtype=torch.float32)
            
            yield batch
    
    def __len__(self):
        return len(self.sampler)


def create_data_loader(dataset, batch_size, shuffle=True, use_speaker_disentanglement=False, num_workers=0):
    """Create appropriate data loader based on speaker disentanglement setting"""
    if use_speaker_disentanglement:
        print("🗣️  Using Speaker-Grouped DataLoader")
        return SpeakerGroupedDataLoader(dataset, batch_size, shuffle, num_workers)
    else:
        print("📦 Using Standard DataLoader")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
