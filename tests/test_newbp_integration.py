import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.physical_layer import SpatiallyVaryingPhysicalLayer
from models.aberration_net import AberrationNet
from models.zernike import DifferentiableZernikeGenerator

def test_newbp_integration():
    """
    Test 4: End-to-End Integration Test
    Verify that NewBP works within the full physical layer and allows training.
    """
    print("\n=== Test 4: NewBP Integration & Training Loop ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Setup Models
    aberration_net = AberrationNet(num_coeffs=15, hidden_dim=32, a_max=2.0).to(device)
    zernike_gen = DifferentiableZernikeGenerator(n_modes=15, pupil_size=64, kernel_size=31, device=device).to(device)
    
    # Create two layers: Standard and NewBP
    layer_std = SpatiallyVaryingPhysicalLayer(
        aberration_net=aberration_net,
        zernike_generator=zernike_gen,
        patch_size=32, # Small patch for speed
        stride=16,
        use_newbp=False
    ).to(device)
    
    layer_newbp = SpatiallyVaryingPhysicalLayer(
        aberration_net=aberration_net,
        zernike_generator=zernike_gen,
        patch_size=32,
        stride=16, 
        use_newbp=True
    ).to(device)
    
    # 2. Synthetic Data
    B, C, H, W = 2, 3, 64, 64
    x_input = torch.randn(B, C, H, W, device=device, requires_grad=True)
    target = torch.randn(B, C, H, W, device=device) # Dummy target
    
    # 3. Test Standard Forward/Backward
    print("\n--- Testing Standard Implementation ---")
    optimizer_std = optim.SGD(list(layer_std.parameters()) + [x_input], lr=0.01)
    optimizer_std.zero_grad()
    
    y_std = layer_std(x_input)
    loss_std = nn.MSELoss()(y_std, target)
    loss_std.backward()
    
    print(f"Forward Output Shape: {y_std.shape}")
    print(f"Loss: {loss_std.item():.6f}")
    if x_input.grad is not None:
        print(f"Input Gradient Norm: {x_input.grad.norm().item():.6f}")
    else:
        print("[FAILURE] Input gradient missing!")
        
    # Reset grads
    x_input.grad = None
    optimizer_std.zero_grad()
    
    # 4. Test NewBP Forward/Backward
    print("\n--- Testing NewBP Implementation ---")
    
    # Ensure same weights for fair comparison (sharing same net instances)
    # Note: aberration_net and zernike_gen are shared
    
    y_newbp = layer_newbp(x_input)
    
    # Check forward consistency (should be identical up to float precision)
    diff = (y_std - y_newbp).abs().max().item()
    print(f"Forward Pass Difference (Std vs NewBP): {diff:.6e}")
    if diff < 1e-5:
        print("[SUCCESS] Forward pass matches Standard Implementation")
    else:
        print("[FAILURE] Forward pass mismatch!")
        
    loss_newbp = nn.MSELoss()(y_newbp, target)
    loss_newbp.backward()
    
    print(f"Loss: {loss_newbp.item():.6f}")
    
    if x_input.grad is not None:
        grad_norm = x_input.grad.norm().item()
        print(f"Input Gradient Norm: {grad_norm:.6f}")
        if grad_norm > 0:
             print("[SUCCESS] NewBP Backward Pass successful (Gradient flow confirmed)")
    else:
        print("[FAILURE] NewBP Input gradient missing!")
        
    # Check parameter gradients
    param_grad_norm = sum(p.grad.norm().item() for p in layer_newbp.parameters() if p.grad is not None)
    print(f"Parameter Gradient Norm sum: {param_grad_norm:.6f}")
    if param_grad_norm > 0:
        print("[SUCCESS] Gradients propagated to AberrationNet parameters")
        
    print("\n=== Integration Test Complete ===")

if __name__ == "__main__":
    # Redirect stdout to file
    os.makedirs('results', exist_ok=True)
    with open('results/test_integration.log', 'w') as f:
        sys.stdout = f
        test_newbp_integration()
    sys.stdout = sys.__stdout__
    print("Test finished. Results written to results/test_integration.log")
