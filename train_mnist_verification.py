
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from models.zernike import DifferentiableZernikeGenerator
from models.aberration_net import PolynomialAberrationNet, AberrationNet
from models.restoration_net import RestorationNet
from models.physical_layer import SpatiallyVaryingPhysicalLayer
from trainer import DualBranchTrainer

# =============================================================================
# 1. Dataset Class: MNISTMosaicDataset
# =============================================================================
class MNISTMosaicDataset(Dataset):
    """
    Creates a large canvas (e.g., 256x256) by tiling multiple MNIST digits.
    Expands 1-channel MNIST to 3-channel RGB.
    """
    def __init__(self, root='./data', train=True, download=True, canvas_size=256, grid_size=8):
        self.canvas_size = canvas_size
        self.grid_size = grid_size
        self.digit_size = 28
        
        # Load standard MNIST
        # We don't apply transforms here, we want raw PIL images or Tensors to paste
        self.mnist_data = datasets.MNIST(
            root=root, 
            train=train, 
            download=download,
            transform=transforms.ToTensor() # Returns [1, 28, 28] in [0, 1]
        )
        
        # Pre-calculate grid positions to center the grid on canvas
        # 8 * 28 = 224 pixels used. 256 - 224 = 32 pixels margin.
        total_content_size = grid_size * self.digit_size
        start_offset = (canvas_size - total_content_size) // 2
        
        self.positions = []
        for r in range(grid_size):
            for c in range(grid_size):
                y = start_offset + r * self.digit_size
                x = start_offset + c * self.digit_size
                self.positions.append((y, x))
        
        self.num_digits_per_image = len(self.positions)
        
    def __len__(self):
        # We can define an arbitrary epoch length or map 1:1 to MNIST length
        # Let's map 1:1 but each sample constructs a random mosaic
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # Create blank black canvas (C, H, W)
        canvas = torch.zeros((3, self.canvas_size, self.canvas_size), dtype=torch.float32)
        
        # Fill the grid with random digits
        # We use the 'idx' for the first digit to ensure some determinism/coverage if iterating,
        # but pick others randomly.
        
        indices = torch.randint(0, len(self.mnist_data), (self.num_digits_per_image,))
        indices[0] = idx # Ensure we visit the dataset index
        
        for i, pos in enumerate(self.positions):
            y, x = pos
            digit_idx = int(indices[i].item())
            digit_img, _ = self.mnist_data[digit_idx] # [1, 28, 28]
            
            # Paste into canvas (replicate to 3 channels)
            # digit_img is [0, 1]
            canvas[:, y:y+self.digit_size, x:x+self.digit_size] = digit_img.repeat(3, 1, 1)
            
        return canvas

# =============================================================================
# 2. Physical Blur Generator
# =============================================================================
class PhysicalBlurGenerator:
    """
    Wraps a SpatiallyVaryingPhysicalLayer with fixed GT aberrations
    to generate blurred images on-the-fly.
    """
    def __init__(self, config, device):
        self.device = device
        
        # 1. Zernike Generator
        self.zernike_gen = DifferentiableZernikeGenerator(
            n_modes=config.physics.n_modes,
            pupil_size=config.physics.pupil_size,
            kernel_size=config.physics.kernel_size,
            oversample_factor=config.physics.oversample_factor,
            wavelengths=config.physics.wavelengths,
            ref_wavelength=config.physics.ref_wavelength,
            device=device
        ).to(device)
        
        # 2. GT Aberration Net (Polynomial)
        # Randomly initialized to create spatially varying blur
        self.aberration_net = PolynomialAberrationNet(
            n_coeffs=config.physics.n_modes,
            degree=config.aberration_net.polynomial.degree,
            a_max=config.aberration_net.a_max
        ).to(device)
        
        # Random initialization for interesting blur
        # std=0.5 provides reasonable aberrations
        nn.init.normal_(self.aberration_net.poly_weights, mean=0.0, std=0.5)
        
        # Freeze weights
        for param in self.aberration_net.parameters():
            param.requires_grad = False
            
        # 3. Physical Layer (using GT components)
        self.physical_layer = SpatiallyVaryingPhysicalLayer(
            aberration_net=self.aberration_net,
            zernike_generator=self.zernike_gen,
            patch_size=config.ola.patch_size,
            stride=config.ola.stride,
            pad_to_power_2=config.ola.pad_to_power_2,
            use_newbp=config.ola.use_newbp
        ).to(device)
        
        self.physical_layer.eval() # Ensure eval mode (though no BN/Dropout usually)

    def generate(self, x_clean):
        """
        x_clean: [B, 3, H, W]
        Returns: y_blurred [B, 3, H, W] (with noise)
        """
        with torch.no_grad():
            # 1. Physical Convolution
            y_blurred = self.physical_layer(x_clean)
            
            # 2. Sensor Noise (Gaussian)
            sigma = 0.01
            noise = torch.randn_like(y_blurred) * sigma
            y_blurred = y_blurred + noise
            
            # 3. Clamp
            y_blurred = torch.clamp(y_blurred, 0.0, 1.0)
            
        return y_blurred

# =============================================================================
# 3. Main Training Script
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='MNIST Verification for Defocus Project')
    parser.add_argument('--config', type=str, default='config/mnist_debug.yaml', help='Path to config file')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("MNIST Verification: Physics-Driven Blind Deconvolution")
    print("="*60)
    
    # 1. Load Config
    config = load_config(args.config)
    device = config.experiment.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = 'cpu'
    print(f"Device: {device}")
    
    # 2. Prepare Data
    print("\n[Step 1] Preparing MNIST Mosaic Dataset...")
    dataset = MNISTMosaicDataset(
        root='./data', 
        train=True, 
        download=True, 
        canvas_size=config.data.image_height,
        grid_size=8 # Fits 8x28=224 inside 256
    )
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=True, num_workers=config.data.num_workers)
    
    # Verify Data Shape
    dummy_x = next(iter(dataloader))
    print(f"  Dataset Sample Shape: {dummy_x.shape} (Range: [{dummy_x.min():.2f}, {dummy_x.max():.2f}])")
    
    # 3. Prepare Blur Generator (GT Simulator)
    print("\n[Step 2] Initializing Physical Blur Generator (GT Optics)...")
    blur_generator = PhysicalBlurGenerator(config, device)
    
    # Verify Blur Generation
    dummy_x = dummy_x.to(device)
    dummy_y = blur_generator.generate(dummy_x)
    print(f"  Blurred Sample Shape: {dummy_y.shape}")
    
    # 4. Prepare Training Models (Restoration + Est Physical Layer)
    print("\n[Step 3] Initializing Restoration Models...")
    
    # Zernike Generator (Shared logic, new instance to be safe or reuse)
    # Reusing the class logic but creating fresh instance for training setup
    train_zernike_gen = DifferentiableZernikeGenerator(
        n_modes=config.physics.n_modes,
        pupil_size=config.physics.pupil_size,
        kernel_size=config.physics.kernel_size,
        oversample_factor=config.physics.oversample_factor,
        wavelengths=config.physics.wavelengths,
        ref_wavelength=config.physics.ref_wavelength,
        device=device
    ).to(device)
    
    # Estimated Aberration Net (Initialized to near-zero)
    est_aberration_net = PolynomialAberrationNet(
        n_coeffs=config.aberration_net.n_coeffs,
        degree=config.aberration_net.polynomial.degree,
        a_max=config.aberration_net.a_max
    ).to(device)
    # Init to near zero
    nn.init.normal_(est_aberration_net.poly_weights, mean=0.0, std=0.01)
    
    # Estimated Physical Layer
    est_physical_layer = SpatiallyVaryingPhysicalLayer(
        aberration_net=est_aberration_net,
        zernike_generator=train_zernike_gen,
        patch_size=config.ola.patch_size,
        stride=config.ola.stride,
        pad_to_power_2=config.ola.pad_to_power_2,
        use_newbp=config.ola.use_newbp
    ).to(device)

    # Restoration Net
    restoration_net = RestorationNet(
        n_channels=config.restoration_net.n_channels,
        n_classes=config.restoration_net.n_classes,
        base_filters=config.restoration_net.base_filters,
        bilinear=config.restoration_net.bilinear,
        use_coords=config.restoration_net.use_coords
    ).to(device)
    
    # 5. Trainer
    trainer = DualBranchTrainer(
        restoration_net=restoration_net,
        physical_layer=est_physical_layer,
        lr_restoration=config.training.optimizer.lr_restoration,
        lr_optics=config.training.optimizer.lr_optics,
        lambda_sup=config.training.loss.lambda_sup, # 0.0 for self-supervised mostly, but we can enable if we trust X_gt
        lambda_coeff=config.training.loss.lambda_coeff,
        lambda_smooth=config.training.loss.lambda_smooth,
        device=device
    )
    
    # 6. Training Loop
    print("\n[Step 4] Starting Training Loop...")
    epochs = config.experiment.epochs
    
    # Create results dir
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    
    # Prepare fixed validation sample for consistent visualization
    fixed_val_x = next(iter(dataloader)).to(device)
    with torch.no_grad():
        fixed_val_y = blur_generator.generate(fixed_val_x)

    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        steps = 0
        
        for i, x_clean in enumerate(dataloader):
            x_clean = x_clean.to(device)
            
            # Generate Synthetic Blur
            with torch.no_grad():
                y_blurred = blur_generator.generate(x_clean)
                
            # Train Step
            # Option A: Self-supervised (Reblur(Restored) vs Blurred)
            # Option B: Supervised (Restored vs Clean) - Since we have GT, we could use it!
            # Let's stick to the prompt description: 
            # "x_gt -> gt_physical -> y_blurred -> restoration_net -> x_pred"
            # And usually in real blind decon, we assume we don't have x_gt. 
            # But for verification, we can calculate loss against x_gt to see if it learns?
            # Or stick to the project's 'demo_train.py' logic which uses self-supervised likely.
            # However, the user prompt said: "(optional) feed y_blurred to est_physical_layer for self-supervised".
            # To ensure it learns quickly for this debug, maybe we can cheat and use supervised loss?
            # Or just use the trainer as is. The trainer likely implements self-supervised loss if lambda_sup=0.
            
            # Use self-supervised training as per default config
            stats = trainer.train_step(y_blurred, X_gt=x_clean) 
            # If you want to debug with supervised loss to verify net capacity:
            # stats = trainer.train_step(y_blurred, X_gt=x_clean) 
            
            epoch_loss += stats['loss']
            steps += 1
            
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}] Loss: {stats['loss']:.4f} (Smooth: {stats['loss_smooth']:.4f})")
                
            # if i >= 50: # Limit steps per epoch for quick debug
            #     break
        
        avg_loss = epoch_loss / steps
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")

        # 7. Validation & Visualization
        print(f"\n[Step 5] Saving Visualization Results (Epoch {epoch+1})...")
        
        # Pick one sample
        val_clean = fixed_val_x[0:1] # [1, 3, H, W]
        val_blurred = fixed_val_y[0:1] # [1, 3, H, W]
        
        with torch.no_grad():
            restoration_net.eval()
            val_pred = restoration_net(val_blurred)
            val_pred = torch.clamp(val_pred, 0, 1)
            # Switch back to train mode!
            restoration_net.train()
            
        # Convert to numpy for plotting
        def to_np(t):
            return t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
        img_clean = to_np(val_clean)
        img_blur = to_np(val_blurred)
        img_pred = to_np(val_pred)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_clean)
        axes[0].set_title('GT Clean (Mosaic)')
        axes[1].imshow(img_blur)
        axes[1].set_title('GT Blurred (Simulated)')
        axes[2].imshow(img_pred)
        axes[2].set_title(f'Restoration (Epoch {epoch+1})')
        
        for ax in axes: ax.axis('off')
        
        save_path = os.path.join(config.experiment.output_dir, f'comparison_epoch{epoch+1}.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig) # Close figure to free memory
        print(f"Saved comparison to: {save_path}")

    print("MNIST Verification Pipeline Completed.")

if __name__ == '__main__':
    main()
