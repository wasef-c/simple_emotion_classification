# Version B Configuration Files

This directory contains pre-configured experiment setups for different research scenarios.

## Available Configurations

### 🔧 **baseline.py**
- **Purpose**: Baseline without curriculum learning
- **Settings**: 
  - No curriculum learning
  - Basic model (1024 hidden, 0.3 dropout)
  - Both LOSO and cross-corpus evaluation
  - 20 epochs
- **Use Case**: Baseline comparison for curriculum learning studies

### 📚 **curriculum_linear.py**
- **Purpose**: Linear curriculum pacing
- **Settings**:
  - 8 epochs of linear curriculum progression
  - Steady sample introduction rate
  - 25 total epochs
- **Use Case**: Standard curriculum learning approach

### 📈 **curriculum_sqrt.py**
- **Purpose**: Square root curriculum pacing
- **Settings**:
  - 10 epochs of sqrt curriculum progression
  - Fast initial growth, slower later
  - 25 total epochs
- **Use Case**: Aggressive early curriculum, gentle later

### ⚡ **cross_corpus_only.py**
- **Purpose**: Fast cross-corpus evaluation only
- **Settings**:
  - No LOSO (much faster)
  - Larger batch size (48)
  - 15 epochs only
  - Trains on IEMOCAP → tests on MSPI & MSPP
- **Use Case**: Quick experiments, model development

### 🏫 **loso_only.py**
- **Purpose**: Traditional academic validation
- **Settings**:
  - LOSO evaluation only
  - Conservative learning rate
  - Trains on IEMOCAP
- **Use Case**: Academic papers, traditional validation

### 🔬 **ablation_study.py**
- **Purpose**: Comprehensive ablation study
- **Settings**:
  - Both evaluation modes
  - Optimized hyperparameters
  - Refined VAD values based on literature
- **Use Case**: Final comparison studies

## Usage

### Method 1: Direct Config Runner
```bash
cd "Version B"
python run_config.py baseline
python run_config.py curriculum_linear
python run_config.py cross_corpus_only
```

### Method 2: Manual Config Loading
```python
# In your script
from configs.baseline import get_config
config = get_config()
# Use config...
```

### Method 3: Modify and Create New Configs
```python
# Copy existing config and modify
from configs.baseline import get_config

def get_my_config():
    config = get_config()
    config.batch_size = 64
    config.num_epochs = 30
    config.experiment_name = "my_custom_experiment"
    return config
```

## Configuration Parameters

### Core Settings
- `batch_size`: Training batch size
- `num_epochs`: Total training epochs
- `learning_rate`: Learning rate
- `weight_decay`: L2 regularization

### Model Settings  
- `hidden_dim`: Hidden layer dimension
- `dropout`: Dropout probability

### Dataset Settings
- `train_dataset`: "MSPI", "IEMO", or "MSPP"
- `evaluation_mode`: "loso", "cross_corpus", or "both"
- `val_split`: Validation split for cross-corpus mode

### Curriculum Learning
- `use_curriculum_learning`: Enable/disable curriculum
- `curriculum_epochs`: Number of curriculum epochs
- `curriculum_pacing`: "linear", "sqrt", or "log"
- `difficulty_method`: "euclidean_distance"
- `expected_vad`: Expected VAD values per emotion class

### WandB Settings
- `wandb_project`: WandB project name
- `experiment_name`: Experiment identifier

## Quick Start Examples

**Run baseline comparison:**
```bash
python run_config.py baseline
```

**Fast development iteration:**
```bash
python run_config.py cross_corpus_only
```

**Full academic evaluation:**
```bash
python run_config.py loso_only
```

**Comprehensive study:**
```bash
python run_config.py ablation_study
```

## Tips

1. **Start with `cross_corpus_only`** for quick iterations
2. **Use `baseline`** to verify curriculum learning helps
3. **Run `ablation_study`** for final comparisons
4. **Create custom configs** by copying and modifying existing ones
5. **Check WandB project names** to avoid mixing experiments