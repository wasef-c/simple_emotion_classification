#!/usr/bin/env python3
"""
Simple emotion recognition model
"""

import torch
import torch.nn as nn


# # class SimpleEmotionClassifier(nn.Module):
#     """Simple feedforward classifier for emotion recognition"""
    
#     def __init__(self, input_dim=768, hidden_dim=1024, num_classes=4, dropout=0.3):
#         super().__init__()
        
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_classes)
#         )
    
#     def forward(self, x):
#         # Handle both 2D and 3D inputs
#         if len(x.shape) == 3:
#             # x shape: (batch_size, sequence_length, input_dim)
#             # Global average pooling
#             x = x.mean(dim=1)  # (batch_size, input_dim)
#         # If already 2D (batch_size, input_dim), use as-is
        
#         # Classification
#         logits = self.classifier(x)  # (batch_size, num_classes)
#         return logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_proj = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        # Compute attention scores
        scores = self.attention_proj(x).squeeze(-1)  # (batch_size, seq_len)
        weights = F.softmax(scores, dim=1)           # (batch_size, seq_len)
        
        # Weighted sum of x with attention weights
        weighted_sum = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)
        return weighted_sum

class SimpleEmotionClassifier(nn.Module):
    """Simple feedforward classifier with attention pooling for emotion recognition"""
    
    def __init__(self, input_dim=768, hidden_dim=1024, num_classes=4, dropout=0.3):
        super().__init__()
        
        self.attention_pooling = AttentionPooling(input_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Handle both 2D and 3D inputs
        if len(x.shape) == 3:
            # x shape: (batch_size, sequence_length, input_dim)
            # Use attention pooling instead of average pooling
            x = self.attention_pooling(x)  # (batch_size, input_dim)
        # If already 2D (batch_size, input_dim), use as-is
        
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits


