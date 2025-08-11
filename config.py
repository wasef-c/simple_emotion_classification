#!/usr/bin/env python3
"""
Simple configuration class for emotion recognition
"""

class Config:
    """Configuration class with basic parameters"""
    
    def __init__(self):
        # Data settings
        self.batch_size = 32
        self.num_epochs = 20
        
        # Model settings
        self.hidden_dim = 1024
        self.dropout = 0.3
        self.num_classes = 4  # neutral, happy, sad, anger
        
        # Training settings
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        # Dataset settings
        self.train_dataset = "MSPI"  # or "IEMO or MSPP
        
        # Evaluation settings
        self.evaluation_mode = "both"  # "loso", "cross_corpus", "both"
        self.val_split = 0.2  # validation split for cross-corpus only mode
        
        # Curriculum Learning settings
        self.use_curriculum_learning = True
        self.curriculum_epochs = 10  # number of epochs to gradually introduce samples
        self.curriculum_pacing = "linear"  # or "sqrt", "log"
        self.difficulty_method = "euclidean_distance"  # method to calculate difficulty
        self.curriculum_type = "difficulty"  # "difficulty" or "preset_order"
        
        # Expected VAD values for difficulty calculation
        self.expected_vad = {
            0: [3.0, 3.0, 3.0],  # neutral
            1: [4.2, 3.8, 3.5],  # happy
            2: [1.8, 2.2, 2.5],  # sad
            3: [2.5, 4.0, 3.2]   # anger
        }
        
        # WandB settings
        self.wandb_project = "Full_Ablation"
        self.experiment_name = "baseline"
        
        # Class labels
        self.class_names = ["neutral", "happy", "sad", "anger"]
        self.use_difficulty_scaling = False
        
        # Speaker disentanglement
        self.use_speaker_disentanglement = False