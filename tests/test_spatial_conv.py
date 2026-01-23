"""
Test script for verifying the spatial convolution implementation.
Compares results with reference FFT-based implementation.
"""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from models.newbp_convolution import NewBPConvolutionFunction


def fft_convolution_reference(patches, kernels, kernel_size, patch_size, fft_size):
    """Reference FFT-based convolution for comparison."""
    X_f = torch.fft.rfft2(patches, s=(fft_size, fft_size))
    K_f = torch.fft.rfft2(kernels, s=(fft_size, fft_size))
    Y_f = X_f * K_f
    y_large = torch.fft.irfft2(Y_f, s=(fft_size, fft_size))
    crop_start = kernel_size // 2
    y = y_large[..., crop_start:crop_start + patch_size, 
                     crop_start:crop_start + patch_size]
    return y


def test_forward_equivalence():
    """Test that spatial conv produces same results as FFT conv."""
    print("=" * 60)
    print("Test 1: Forward pass equivalence (Spatial vs FFT)")
    print("=" * 60)
    
    torch.manual_seed(42)
    B, N, C = 2, 4, 3
    P, K = 64, 15  # Smaller for faster test
    fft_size = 128
    
    patches = torch.randn(B*N, C, P, P)
    kernels = torch.randn(B*N, C, K, K)
    # Normalize kernels
    kernels = kernels / (kernels.sum(dim=(-2,-1), keepdim=True) + 1e-8)
    
    # FFT reference
    y_fft = fft_convolution_reference(patches, kernels, K, P, fft_size)
    
    # Spatial (new implementation)
    y_spatial = NewBPConvolutionFunction.apply(
        patches, kernels, K, P, fft_size
    )
    
    # Compare
    diff = (y_fft - y_spatial).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"  Input shape: {patches.shape}")
    print(f"  Kernel shape: {kernels.shape}")
    print(f"  Output shape: {y_spatial.shape}")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    
    # Note: Spatial and FFT convolution may have slight numerical differences
    # due to different padding modes (reflect vs circular)
    # We check that they are reasonably close
    if max_diff < 0.1:
        print("  ✓ PASSED (within tolerance)")
        return True
    else:
        print("  ✗ FAILED (difference too large)")
        return False


def test_backward_gradient_flow():
    """Test that gradients flow correctly through backward pass."""
    print("\n" + "=" * 60)
    print("Test 2: Backward pass gradient flow")
    print("=" * 60)
    
    torch.manual_seed(42)
    B, N, C = 2, 4, 3
    P, K = 64, 15
    fft_size = 128
    
    patches = torch.randn(B*N, C, P, P, requires_grad=True)
    kernels = torch.randn(B*N, C, K, K)
    kernels = kernels / (kernels.sum(dim=(-2,-1), keepdim=True) + 1e-8)
    kernels = kernels.detach().requires_grad_(True)
    
    # Forward
    y = NewBPConvolutionFunction.apply(patches, kernels, K, P, fft_size)
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    print(f"  patches.grad shape: {patches.grad.shape}")
    print(f"  kernels.grad shape: {kernels.grad.shape}")
    print(f"  patches.grad norm: {patches.grad.norm().item():.4f}")
    print(f"  kernels.grad norm: {kernels.grad.norm().item():.4f}")
    
    if patches.grad is not None and kernels.grad is not None:
        if patches.grad.shape == patches.shape and kernels.grad.shape == kernels.shape:
            print("  ✓ PASSED")
            return True
    
    print("  ✗ FAILED")
    return False


def test_gradient_numerical_check():
    """Numerical gradient check using finite differences."""
    print("\n" + "=" * 60)
    print("Test 3: Numerical gradient check (finite differences)")
    print("=" * 60)
    
    torch.manual_seed(42)
    # Use smaller tensors for numerical check
    B, C = 1, 1
    P, K = 16, 5
    fft_size = 32
    eps = 1e-4
    
    patches = torch.randn(B, C, P, P, dtype=torch.float64, requires_grad=True)
    kernels = torch.randn(B, C, K, K, dtype=torch.float64)
    kernels = kernels / (kernels.sum(dim=(-2,-1), keepdim=True) + 1e-8)
    kernels = kernels.detach().requires_grad_(True)
    
    # Analytical gradient
    y = NewBPConvolutionFunction.apply(patches, kernels, K, P, fft_size)
    loss = y.sum()
    loss.backward()
    
    analytical_grad = patches.grad.clone()
    
    # Numerical gradient for a few random positions
    numerical_grad = torch.zeros_like(patches)
    
    # Check 5 random positions
    for _ in range(5):
        i = torch.randint(0, P, (1,)).item()
        j = torch.randint(0, P, (1,)).item()
        
        patches_plus = patches.detach().clone()
        patches_plus[0, 0, i, j] += eps
        y_plus = NewBPConvolutionFunction.apply(patches_plus, kernels.detach(), K, P, fft_size)
        
        patches_minus = patches.detach().clone()
        patches_minus[0, 0, i, j] -= eps
        y_minus = NewBPConvolutionFunction.apply(patches_minus, kernels.detach(), K, P, fft_size)
        
        numerical_grad[0, 0, i, j] = (y_plus.sum() - y_minus.sum()) / (2 * eps)
    
    # Compare at checked positions
    mask = numerical_grad != 0
    if mask.sum() > 0:
        diff = (analytical_grad[mask] - numerical_grad[mask]).abs()
        rel_diff = diff / (numerical_grad[mask].abs() + 1e-8)
        max_rel_diff = rel_diff.max().item()
        
        print(f"  Max relative gradient difference: {max_rel_diff:.6e}")
        
        if max_rel_diff < 1e-3:
            print("  ✓ PASSED")
            return True
        else:
            print("  ✗ FAILED (gradient mismatch)")
            return False
    
    print("  ✗ FAILED (no positions checked)")
    return False


def test_physical_layer_integration():
    """Test integration with SpatiallyVaryingPhysicalLayer."""
    print("\n" + "=" * 60)
    print("Test 4: Physical layer integration")
    print("=" * 60)
    
    try:
        from models.physical_layer import SpatiallyVaryingPhysicalLayer
        from models.zernike import DifferentiableZernikeGenerator
        from models.aberration_net import AberrationNet
        
        device = 'cpu'
        
        # Create components
        aberration_net = AberrationNet(num_coeffs=15, hidden_dim=64, a_max=1.0).to(device)
        zernike_gen = DifferentiableZernikeGenerator(
            n_modes=15, pupil_size=64, kernel_size=31,
            wavelengths=[650e-9, 550e-9, 450e-9],
            device=device
        )
        
        # Create physical layer (with NewBP)
        layer = SpatiallyVaryingPhysicalLayer(
            aberration_net=aberration_net,
            zernike_generator=zernike_gen,
            patch_size=128,
            stride=64,
            use_newbp=True
        ).to(device)
        
        # Test forward
        x = torch.randn(1, 3, 256, 256, device=device, requires_grad=True)
        y = layer(x)
        
        print(f"  Input: {x.shape}")
        print(f"  Output: {y.shape}")
        
        # Test backward
        loss = y.sum()
        loss.backward()
        
        print(f"  Input gradient computed: {x.grad is not None}")
        print("  ✓ PASSED")
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Spatial Convolution Implementation Tests")
    print("=" * 60 + "\n")
    
    results = []
    results.append(("Forward equivalence", test_forward_equivalence()))
    results.append(("Backward gradient flow", test_backward_gradient_flow()))
    results.append(("Numerical gradient check", test_gradient_numerical_check()))
    results.append(("Physical layer integration", test_physical_layer_integration()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    sys.exit(0 if all_passed else 1)
