import torch
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.newbp_convolution import compute_jacobian_structure, NewBPConvolutionFunction

def test_jacobian_structure():
    """
    Test 1: Verify Jacobian matrix is non-diagonal and has correct structure.
    """
    print("\n=== Test 1: Jacobian Matrix Structure Analysis ===")
    
    # Setup
    K_size = 5
    image_size = 16
    
    # Create a simple Gaussian-like kernel
    kernel = torch.zeros(K_size, K_size)
    center = K_size // 2
    sigma = 1.0
    for i in range(K_size):
        for j in range(K_size):
            y, x = i - center, j - center
            kernel[i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
    kernel = kernel / kernel.sum()
    
    # Compute explicit Jacobian
    J = compute_jacobian_structure(kernel, image_size)
    
    # Analysis
    diagonal_elements = torch.diagonal(J)
    off_diagonal_mask = ~torch.eye(J.shape[0], dtype=torch.bool)
    off_diagonal_elements = J[off_diagonal_mask]
    
    max_diag = diagonal_elements.max().item()
    max_off_diag = off_diagonal_elements.max().item()
    
    print(f"Jacobian Shape: {J.shape}")
    print(f"Max Diagonal Element: {max_diag:.6f}")
    print(f"Max Off-Diagonal Element: {max_off_diag:.6f}")
    
    # Assertions
    if max_off_diag > 1e-6:
        print("[SUCCESS] Jacobian has significant off-diagonal elements (Spatial Crosstalk confirmed)")
    else:
        print("[FAILURE] Jacobian appears diagonal (No Crosstalk modeled)")
        
    # Check band structure (sanity check)
    # The band width should be related to kernel size
    # Visualize if possible (save to file)
    plt.figure(figsize=(10, 8))
    plt.imshow(J[:64, :64].numpy(), cmap='viridis') # Show top-left corner
    plt.colorbar(label='Influence Coefficient')
    plt.title(f'Jacobian Matrix Structure (Top-Left 64x64)\nKernel Size: {K_size}x{K_size}')
    plt.xlabel('Source Pixel Index')
    plt.ylabel('Destination Pixel Index')
    
    save_path = 'results/test_jacobian_structure.png'
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path)
    print(f"[INFO] Jacobian visualization saved to {save_path}")

def test_gradient_decomposition():
    """
    Test 2: Verify gradient decomposition into direct and indirect components.
    """
    print("\n=== Test 2: Gradient Decomposition Analysis ===")
    
    # Setup
    B, C, P = 1, 1, 32
    K = 5
    fft_size = P + K - 1
    fft_size = 2 ** ((fft_size - 1).bit_length())  # Power of 2
    
    patches = torch.randn(B, C, P, P, requires_grad=True)
    kernels = torch.randn(B, C, K, K) # Random kernels
    kernels = kernels / kernels.sum(dim=(-1, -2), keepdim=True) # Normalize
    
    # Forward pass using NewBP function
    y = NewBPConvolutionFunction.apply(patches, kernels, K, P, fft_size)
    
    # Create a gradient signal (loss derivative)
    # Let's say we want to increase the value of the center pixel of output
    grad_output = torch.zeros_like(y)
    grad_output[..., P//2, P//2] = 1.0
    
    # Backward pass
    y.backward(grad_output)
    
    grad_total = patches.grad
    
    # Manual decomposition for verification
    # Direct contribution: grad_output * K[center]
    center_idx = K // 2
    k_center = kernels[..., center_idx, center_idx].view(B, C, 1, 1)
    grad_direct_expected = grad_output * k_center
    
    # Indirect is the rest
    grad_indirect_expected = grad_total - grad_direct_expected
    
    print(f"Total Gradient Norm: {grad_total.norm().item():.6f}")
    print(f"Direct Gradient Norm: {grad_direct_expected.norm().item():.6f}")
    print(f"Indirect Gradient Norm: {grad_indirect_expected.norm().item():.6f}")
    
    # Check if indirect gradients exist
    if grad_indirect_expected.norm().item() > 0:
         print("[SUCCESS] Indirect gradients detected (Neighboring pixels receive gradients)")
    else:
         print("[FAILURE] No indirect gradients found")

if __name__ == "__main__":
    # Redirect stdout to file
    os.makedirs('results', exist_ok=True)
    with open('results/test_jacobian.log', 'w') as f:
        sys.stdout = f
        test_jacobian_structure()
        test_gradient_decomposition()
    sys.stdout = sys.__stdout__
    print("Test finished. Results written to results/test_jacobian.log")
