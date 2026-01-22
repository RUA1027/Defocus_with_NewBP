import torch
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add root to python path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from demo_train import build_models_from_config, build_trainer_from_config
from utils import DPDDDataset

def test_pipeline():
    print("=== DPDD Single Image Pipeline Test ===")
    
    # 1. Load Config
    config_path = "config/default.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return
        
    print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Force settings for this test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    
    # Override batch size to 1 for single image test
    # (Checking if models handle batch=1 correctly usually implies they handle N)
    print("Overriding batch_size to 1 for test...")
    config.data.batch_size = 1
    config.experiment.device = device

    # 2. Build Dataset
    print("\nInitializing Dataset...")
    if not os.path.exists(config.data.data_root):
        print(f"Error: Data root {config.data.data_root} does not exist.")
        print("Please run 'preprocess_dpdd.py' first.")
        return

    # Use 'train' set for testing
    dataset = DPDDDataset(
        root_dir=config.data.data_root, 
        mode='train', 
        transform=None # Default ToTensor
    )
    
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return
        
    print(f"Dataset loaded. Size: {len(dataset)}")
    
    # Create Loader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Get 1 batch
    try:
        # returns (blur, sharp)
        blur_img, sharp_img = next(iter(loader))
        print(f"Successfully loaded one batch.")
        print(f"Blur Shape: {blur_img.shape}")
        print(f"Sharp Shape: {sharp_img.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Build Models & Trainer
    print("\nBuilding Models...")
    try:
        zernike_gen, aberration_net, restoration_net, physical_layer = \
            build_models_from_config(config, device)
        
        trainer = build_trainer_from_config(config, restoration_net, physical_layer, device)
        print("Models and Trainer built successfully.")
    except Exception as e:
        print(f"Error building models: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Run Training Step
    print("\nRunning Forward/Backward Pass (Train Step)...")
    
    try:
        blur_img = blur_img.to(device)
        sharp_img = sharp_img.to(device)
        
        # train_step(Y, X_gt) -> Y=blur, X_gt=sharp
        metrics = trainer.train_step(blur_img, sharp_img)
        
        print("\n=== Success! ===")
        print("Metrics returned:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")
            
    except RuntimeError as e:
        print(f"RuntimeError during training step: {e}")
        if "out of memory" in str(e):
            print("Hint: GPU Out of Memory. reduce batch_size or image size.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error during training step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
