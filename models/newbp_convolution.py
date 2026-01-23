"""
NewBP (New Backpropagation) Custom Autograd Function for Defocus Deblurring

This module implements a custom autograd function that handles spatial energy crosstalk
during backpropagation, using non-diagonal Jacobian matrices to model how light from
one pixel spreads to neighboring pixels through the Point Spread Function (PSF).

Mathematical Foundation:
-----------------------
Forward: Y = H·X, where H is the crosstalk matrix (convolution with PSF)
Backward: ∂L/∂X = H^T · (∂L/∂Y)

Gradient Decomposition (NewBP):
  ∂L/∂X[i,j] = G_direct + G_indirect
  
  G_direct   = ∂L/∂Y[i,j] × K[0,0]           # Self-contribution (Diagonal elements)
  G_indirect = Σ_{(m,n)≠(0,0)} ∂L/∂Y[m,n] × K[i-m, j-n]  # Neighbor contributions (Off-diagonal)

For circular PSF: K_flipped = K (symmetric), so the backward convolution uses the same kernel.

Reference: 科研日志.md lines 88-145 (NewBP视角下的具体梯度解析)

Implementation Note (2026-01-23):
---------------------------------
Changed from FFT-based frequency domain multiplication to spatial domain convolution
using F.conv2d for better GPU performance on small kernels (K <= 33).
cuDNN provides optimized algorithms (Winograd, im2col) that outperform FFT approach
when kernel size is small relative to image size.
"""

'''
输入数据流 (空域卷积实现)
────────────────────────────────────────────
清晰图像 X [B*N, 3, 128, 128]
  ↓ (reflect 填充 K//2)
X_padded [B*N, 3, 143, 143]  (for K=31)
  ↓ 
Grouped Conv2d (groups=B*N*C)
  ↓
输出 Y [B*N, 3, 128, 128] ✓

PSF 核 K [B*N, 3, 31, 31]
  ↓ reshape
K_grouped [B*N*3, 1, 31, 31]

────────────────────────────────────────────
反向梯度流 (空域卷积实现)
────────────────────────────────────────────
下游梯度 ∂L/∂Y [B*N, 3, 128, 128]
  ↓ (constant 填充 K//2)
  ↓ conv2d with flip(K)
∂L/∂X [B*N, 3, 128, 128] ✓

对核梯度:
X.unfold(K) → [B*N, C*K*K, P*P]
  ↓ bmm with ∂L/∂Y.flatten
∂L/∂K [B*N, 3, 31, 31] ✓
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math


class NewBPConvolutionFunction(torch.autograd.Function):
    """
    Custom autograd function implementing NewBP-aware convolution.
    
    This function explicitly models the non-diagonal Jacobian matrix that arises
    from PSF-based spatial convolution, decomposing gradients into direct and
    indirect components to properly account for energy crosstalk.
    """
    
    @staticmethod
    def forward(ctx, patches, kernels, kernel_size, patch_size, fft_size):
        """
        Forward pass: FFT-based convolution of patches with PSF kernels.
        
        Args:
            patches: [B*N, C, P, P] - Input image patches
            kernels: [B*N, C_k, K, K] - PSF convolution kernels
            kernel_size: int - Size of PSF kernel (K)
            patch_size: int - Size of patches (P)
            fft_size: int - FFT computation size
            
        Returns:
            y_patches: [B*N, C_out, P, P] - Convolved patches
        """
        # Save input patches for kernel gradient computation
        ctx.save_for_backward(patches, kernels)
        ctx.kernel_size = kernel_size
        ctx.patch_size = patch_size
        ctx.fft_size = fft_size
        
        # Determine output channels
        C = patches.shape[1]
        C_k = kernels.shape[1]
        if C == C_k:
            C_out = C
        elif C == 1 and C_k > 1:
            C_out = C_k
        elif C > 1 and C_k == 1:
            C_out = C
        else:
            raise ValueError(f"Channel mismatch: patches={C}, kernels={C_k}")
        
        ctx.input_channels = C
        ctx.kernel_channels = C_k
        ctx.output_channels = C_out
        
        # =====================================================================
        # Spatial Domain Convolution (GPU-optimized via cuDNN)
        # Replaces FFT-based frequency domain multiplication for better performance
        # on small kernels (K <= 33) where cuDNN excels.
        #
        # IMPORTANT: FFT convolution computes true convolution (kernel flipped),
        # while F.conv2d computes cross-correlation (kernel NOT flipped).
        # To match FFT results, we must flip the kernel before F.conv2d.
        # =====================================================================
        
        BN = patches.shape[0]  # B * N_patches
        
        # Flip kernel to convert correlation to convolution (match FFT behavior)
        kernels_flipped = torch.flip(kernels, dims=[-2, -1])
        
        # Pad input for 'same' output size (kernel_size // 2 on each side)
        # Use 'constant' (zero) padding to match FFT's implicit zero-padding
        pad = kernel_size // 2
        patches_padded = F.pad(patches, (pad, pad, pad, pad), mode='constant', value=0)
        
        # Per-sample convolution: each patch has its own kernel
        # Use groups=BN to apply different kernels to different samples
        # Reshape: [BN, C, H, W] -> [1, BN*C, H, W] for grouped conv
        
        if C == C_k:
            # Case 1: Same channels - direct per-channel convolution
            # Reshape patches: [BN, C, H, W] -> [1, BN*C, H, W]
            patches_grouped = patches_padded.view(1, BN * C, 
                                                   patches_padded.shape[2], 
                                                   patches_padded.shape[3])
            # Reshape flipped kernels: [BN, C, K, K] -> [BN*C, 1, K, K]
            kernels_grouped = kernels_flipped.view(BN * C, 1, kernel_size, kernel_size)
            
            # Grouped convolution: each of BN*C groups has 1 input and 1 output channel
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
            
            # Reshape back: [1, BN*C, P, P] -> [BN, C, P, P]
            y_patches = y_grouped.view(BN, C, patch_size, patch_size)
            
        elif C == 1 and C_k > 1:
            # Case 2: Grayscale input, multi-channel kernel (broadcast)
            # Replicate input to match kernel channels
            patches_expanded = patches_padded.expand(-1, C_k, -1, -1)
            patches_grouped = patches_expanded.reshape(1, BN * C_k,
                                                        patches_padded.shape[2],
                                                        patches_padded.shape[3])
            kernels_grouped = kernels_flipped.view(BN * C_k, 1, kernel_size, kernel_size)
            
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C_k)
            y_patches = y_grouped.view(BN, C_k, patch_size, patch_size)
            
        else:  # C > 1 and C_k == 1
            # Case 3: Multi-channel input, single kernel (broadcast kernel)
            kernels_expanded = kernels_flipped.expand(-1, C, -1, -1)
            patches_grouped = patches_padded.view(1, BN * C,
                                                   patches_padded.shape[2],
                                                   patches_padded.shape[3])
            kernels_grouped = kernels_expanded.reshape(BN * C, 1, kernel_size, kernel_size)
            
            y_grouped = F.conv2d(patches_grouped, kernels_grouped, groups=BN * C)
            y_patches = y_grouped.view(BN, C, patch_size, patch_size)
        
        return y_patches
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: NewBP gradient computation with non-diagonal Jacobian.
        """
        patches, kernels = ctx.saved_tensors
        K = ctx.kernel_size
        P = ctx.patch_size
        fft_size = ctx.fft_size
        C_in = ctx.input_channels
        C_out = ctx.output_channels
        
        # =====================================================================
        # --- 1. Compute Gradient w.r.t Input Patches (dL/dX) ---
        # =====================================================================
        # Jacobian J = ∂Y/∂X is a convolution matrix.
        # J^T corresponds to convolution with 180°-rotated (flipped) kernel.
        # 
        # NewBP Decomposition:
        #   ∂L/∂X[i,j] = G_direct + G_indirect
        #   - G_direct (diagonal):   grad_output[i,j] × K[center, center]
        #   - G_indirect (off-diag): Σ grad_output[m,n] × K_flipped[i-m, j-n]
        # 
        # Since forward used flipped kernel (to match FFT convolution),
        # backward needs the original (un-flipped) kernel for J^T.
        # =====================================================================
        
        BN = grad_output.shape[0]
        
        # Pad grad_output for 'same' convolution
        pad = K // 2
        grad_padded = F.pad(grad_output, (pad, pad, pad, pad), mode='constant', value=0)
        
        # Spatial convolution: grad_output * K (original, not flipped)
        # Because forward used flip(K), backward J^T uses K directly
        if C_out == kernels.shape[1]:
            grad_grouped = grad_padded.view(1, BN * C_out, 
                                            grad_padded.shape[2], grad_padded.shape[3])
            k_grouped = kernels.view(BN * C_out, 1, K, K)
            
            grad_X_grouped = F.conv2d(grad_grouped, k_grouped, groups=BN * C_out)
            grad_patches = grad_X_grouped.view(BN, C_out, P, P)
        else:
            # Handle broadcast cases
            grad_grouped = grad_padded.view(1, BN * C_out,
                                            grad_padded.shape[2], grad_padded.shape[3])
            k_expanded = kernels.expand(-1, C_out, -1, -1) if kernels.shape[1] == 1 else kernels
            k_grouped = k_expanded.reshape(BN * C_out, 1, K, K)
            
            grad_X_grouped = F.conv2d(grad_grouped, k_grouped, groups=BN * C_out)
            grad_patches = grad_X_grouped.view(BN, C_out, P, P)
        
        # Match input channels for backward compatibility
        if grad_patches.shape[1] != C_in:
            if C_in == 1 and grad_patches.shape[1] > 1:
                grad_patches = grad_patches.sum(dim=1, keepdim=True)
            elif C_in > 1 and grad_patches.shape[1] == 1:
                grad_patches = grad_patches.repeat(1, C_in, 1, 1)

        # =====================================================================
        # --- 2. Compute Gradient w.r.t Kernels (dL/dK) ---
        # =====================================================================
        # For Y = X * K (convolution), the kernel gradient is:
        #   dL/dK = X ⊛ dL/dY  (cross-correlation)
        # 
        # In spatial domain, this is equivalent to:
        #   dL/dK[ki, kj] = Σ_{i,j} X[i+ki, j+kj] × dL/dY[i, j]
        # 
        # We use F.conv2d with flipped operands to compute this efficiently.
        # =====================================================================
        
        # Cross-correlation: patches ⊛ grad_output
        # Equivalent to: conv2d(patches, grad_output_as_kernel)
        # But we need per-sample correlation, so we use a loop or unfold trick
        
        # For efficiency, compute using unfold + matmul pattern
        # Unfold patches to extract all K×K windows
        patches_unfolded = F.unfold(patches, kernel_size=K, padding=K//2)  # [BN, C*K*K, P*P]
        
        # Reshape grad_output: [BN, C_out, P, P] -> [BN, C_out, P*P]
        grad_flat = grad_output.view(BN, C_out, -1)  # [BN, C_out, P*P]
        
        # Compute correlation via batch matrix multiply
        # patches_unfolded: [BN, C_in*K*K, P*P]
        # grad_flat: [BN, C_out, P*P]
        # Result should be: [BN, C_k, K, K]
        
        C_k = kernels.shape[1]
        
        if C_in == C_out == C_k:
            # Standard case: same channels throughout
            # Per-channel correlation
            grad_kernels_list = []
            for c in range(C_in):
                # Extract channel c windows: [BN, K*K, P*P]
                patch_c = patches_unfolded[:, c*K*K:(c+1)*K*K, :]
                # grad channel c: [BN, 1, P*P]
                grad_c = grad_flat[:, c:c+1, :]
                # Correlation: [BN, K*K, P*P] × [BN, P*P, 1] -> [BN, K*K, 1]
                corr_c = torch.bmm(patch_c, grad_c.transpose(1, 2))  # [BN, K*K, 1]
                grad_kernels_list.append(corr_c.view(BN, 1, K, K))
            grad_kernels = torch.cat(grad_kernels_list, dim=1)  # [BN, C, K, K]
        else:
            # Broadcast cases: simplified computation
            # For C_in=1, C_k>1 or C_in>1, C_k=1
            # Sum over spatial positions
            patches_unfolded_sum = patches_unfolded.view(BN, C_in, K*K, P*P)
            grad_flat_expanded = grad_flat.view(BN, C_out, 1, P*P)
            
            # [BN, C_in, K*K, P*P] × [BN, C_out, P*P, 1] via einsum
            # Output: [BN, max(C_in, C_out), K*K]
            if C_in == 1:
                corr = torch.einsum('bkp,bcp->bck', patches_unfolded_sum.squeeze(1), grad_flat)
            else:  # C_k == 1
                corr = torch.einsum('bckp,bp->bck', patches_unfolded_sum, grad_flat.squeeze(1))
                corr = corr.sum(dim=1, keepdim=True)  # Sum over input channels
            grad_kernels = corr.view(BN, C_k, K, K)
        
        return grad_patches, grad_kernels, None, None, None


class NewBPSpatialConvolution(nn.Module):
    """
    Wrapper module for NewBP convolution that can be used as a layer.
    
    This module provides a clean interface for using NewBP convolution
    within the physical layer, with optional gradient statistics logging.
    """
    
    def __init__(self, enable_grad_logging=False):
        super().__init__()
        self.enable_grad_logging = enable_grad_logging
        
    def forward(self, patches, kernels, kernel_size, patch_size, fft_size):
        """
        Apply NewBP convolution to patches.
        
        Args:
            patches: [B*N, C, P, P]
            kernels: [B*N, C_k, K, K]
            kernel_size: int
            patch_size: int
            fft_size: int
            
        Returns:
            [B*N, C_out, P, P]
        """
        return NewBPConvolutionFunction.apply(
            patches, kernels, kernel_size, patch_size, fft_size
        )


def compute_jacobian_structure(kernel, image_size):
    """
    Utility function to visualize the Jacobian matrix structure for a given PSF kernel.
    
    This function is used for testing and verification. It computes the explicit
    Jacobian matrix for a small image to verify the non-diagonal structure.
    
    Args:
        kernel: [K, K] - PSF kernel
        image_size: int - Size of test image (should be small, e.g., 16)
        
    Returns:
        jacobian: [N, N] - Explicit Jacobian matrix where N = image_size^2
        
    Example:
        >>> kernel = torch.randn(5, 5)
        >>> J = compute_jacobian_structure(kernel, image_size=16)
        >>> # Visualize: plt.imshow(J.abs())
    """
    N = image_size * image_size
    K = kernel.shape[0]
    
    # Create explicit Jacobian matrix
    jacobian = torch.zeros(N, N)
    
    # ---------------------------------------------------------------------
    # Refactored: Vectorized implementation (O(K^2) loops instead of O(N^2))
    # ---------------------------------------------------------------------
    center = K // 2
    device = jacobian.device # Use same device as target matrix
    
    for ki in range(K):
        for kj in range(K):
            val = kernel[ki, kj]
            # Skip near-zero values for sparse efficiency
            if abs(val) < 1e-12: continue
            
            # Calculate displacement relative to center
            # Original logic: ki = center - di => di = center - ki
            di = center - ki
            dj = center - kj
            
            # --- NewBP Gradient Separation ---
            # 1. Direct Gradient (Diagonal Elements): 
            #    Self-contribution where di=0, dj=0 (idx_i == idx_j)
            # 2. Indirect Gradient (Off-Diagonal Elements):
            #    Crosstalk from neighbors where di!=0 or dj!=0
            is_diagonal = (di == 0) and (dj == 0)
            
            # Calculate valid output pixel range [i, j]
            # Condition: 0 <= i < S AND 0 <= i+di < S
            i_start = max(0, -di)
            i_end = min(image_size, image_size - di)
            
            # Condition: 0 <= j < S AND 0 <= j+dj < S
            j_start = max(0, -dj)
            j_end = min(image_size, image_size - dj)
            
            if i_start < i_end and j_start < j_end:
                # Vectorized index generation
                i_idx = torch.arange(i_start, i_end, device=device)
                j_idx = torch.arange(j_start, j_end, device=device)
                
                # Output indices (Rows of Jacobian): i * W + j
                rows = (i_idx[:, None] * image_size + j_idx[None, :]).flatten()
                
                # Input indices (Cols of Jacobian): (i+di) * W + (j+dj)
                # Note: di*W + dj is constant shift for this kernel element
                cols = rows + (di * image_size + dj)
                
                # Batch assignment
                # If is_diagonal: fills main diagonal (Direct Gradient)
                # If not is_diagonal: fills off-diagonal band (Indirect Gradient)
                jacobian[rows, cols] = val
    
    return jacobian


def analyze_gradient_components(grad_total, grad_output, kernel):
    """
    Decompose gradient into direct and indirect components for analysis.
    
    Args:
        grad_total: Total gradient computed by NewBP
        grad_output: Gradient from downstream
        kernel: PSF kernel used in convolution
        
    Returns:
        dict with keys: 'direct', 'indirect', 'direct_ratio'
    """
    K = kernel.shape[-1]
    center_idx = K // 2
    
    # Direct component: grad_output scaled by center kernel value
    K_center = kernel[..., center_idx:center_idx+1, center_idx:center_idx+1]
    grad_direct = grad_output * K_center
    
    # Indirect component: remainder
    grad_indirect = grad_total - grad_direct
    
    # Statistics
    direct_norm = grad_direct.norm().item()
    indirect_norm = grad_indirect.norm().item()
    total_norm = grad_total.norm().item()
    
    return {
        'grad_direct': grad_direct,
        'grad_indirect': grad_indirect,
        'direct_norm': direct_norm,
        'indirect_norm': indirect_norm,
        'total_norm': total_norm,
        'direct_ratio': direct_norm / (total_norm + 1e-10),
        'indirect_ratio': indirect_norm / (total_norm + 1e-10)
    }
