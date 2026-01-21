import torch
import unittest
from models.aberration_net import AberrationNet
from models.restoration_net import RestorationNet
from models.physical_layer import SpatiallyVaryingPhysicalLayer
from models.zernike import DifferentiableZernikeGenerator

class TestBackwardFlow(unittest.TestCase):
    def test_gradients(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Setup
        z_gen = DifferentiableZernikeGenerator(n_modes=15, pupil_size=64, kernel_size=31, device=device)
        ab_net = AberrationNet(num_coeffs=15, hidden_dim=64, a_max=1.0).to(device)
        res_net = RestorationNet(n_channels=1, n_classes=1, base_filters=64).to(device)
        phy_layer = SpatiallyVaryingPhysicalLayer(ab_net, z_gen, patch_size=128, stride=64).to(device)
        
        # Dummy Input
        x = torch.randn(2, 1, 128, 128, device=device, requires_grad=True)
        target = torch.randn(2, 1, 128, 128, device=device)
        
        # Forward
        x_hat = res_net(x)
        y_hat = phy_layer(x_hat)
        
        loss = torch.mean((y_hat - target)**2)
        
        # Backward
        loss.backward()
        
        # Check W gradients (Restoration Net)
        has_w_grad = False
        for param in res_net.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_w_grad = True
                break
        
        # Check Theta gradients (Aberration Net)
        has_theta_grad = False
        for param in ab_net.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_theta_grad = True
                break
                
        print(f"Has Restoration Gradients: {has_w_grad}")
        print(f"Has Optics Gradients: {has_theta_grad}")
        
        self.assertTrue(has_w_grad, "Restoration Network did not receive gradients")
        self.assertTrue(has_theta_grad, "Aberration Network did not receive gradients")

if __name__ == '__main__':
    unittest.main()
