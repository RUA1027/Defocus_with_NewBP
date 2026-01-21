import torch
import unittest
from models.zernike import DifferentiableZernikeGenerator

class TestPSFNormalization(unittest.TestCase):
    def test_normalization(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        N = 15
        
        # Test batch of coefficients
        batch_size = 10
        coeffs = torch.randn(batch_size, N, device=device)
        
        generator = DifferentiableZernikeGenerator(n_modes=N, device=device)
        
        kernels = generator(coeffs) # [B, 1, K, K]
        
        sums = kernels.sum(dim=(-1, -2))
        
        print(f"Kernel sums: {sums.flatten()}")
        
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5), 
                        "PSF Energy is not conserved (sum != 1)")

if __name__ == '__main__':
    unittest.main()
