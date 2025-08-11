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


class SimpleEmotionClassifier(nn.Module):
    """Simple feedforward classifier with attention pooling for emotion recognition"""
    
    def __init__(self, input_dim=768, hidden_dim=1024, num_classes=4, dropout=0.1):
        super().__init__()
                
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        
        logits = self.classifier(x)  # (batch_size, num_classes)
        return logits

