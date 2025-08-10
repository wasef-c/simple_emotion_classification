#!/usr/bin/env python3
"""
Configuration runner - load and run specific config files
"""

import sys
import importlib.util
from pathlib import Path

def load_config_from_file(config_path):
    """Load configuration from a Python file"""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()

def main():
    """Main runner function"""
    if len(sys.argv) != 2:
        print("Usage: python run_config.py <config_name>")
        print("\nAvailable configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.py"):
                if config_file.name != "__init__.py":
                    print(f"  - {config_file.stem}")
        else:
            print("  No configs directory found!")
        return
    
    config_name = sys.argv[1]
    config_path = Path(f"configs/{config_name}.py")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        print("\nAvailable configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.py"):
                if config_file.name != "__init__.py":
                    print(f"  - {config_file.stem}")
        return
    
    print(f"🔧 Loading config: {config_name}")
    config = load_config_from_file(config_path)
    
    print(f"📊 Experiment: {config.experiment_name}")
    print(f"📚 Dataset: {config.train_dataset}")
    print(f"🎯 Evaluation: {config.evaluation_mode}")
    print(f"📖 Curriculum: {'Enabled' if config.use_curriculum_learning else 'Disabled'}")
    if config.use_curriculum_learning:
        print(f"   - Epochs: {config.curriculum_epochs}")
        print(f"   - Pacing: {config.curriculum_pacing}")
    
    # Import and run main with the loaded config
    from main import main as run_experiment
    
    # Override the default config
    import config as default_config
    original_config = default_config.Config
    default_config.Config = lambda: config
    
    try:
        results = run_experiment()
        print(f"✅ Experiment completed successfully!")
        return results
    finally:
        # Restore original config
        default_config.Config = original_config

if __name__ == "__main__":
    main()