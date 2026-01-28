import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_config
from utils.model_builder import build_models_from_config
from utils.metrics import PerformanceEvaluator


class DummyBlurSharpDataset(Dataset):
    def __init__(self, length=2, height=64, width=64):
        self.length = length
        self.height = height
        self.width = width

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        blur = torch.rand(3, self.height, self.width)
        sharp = torch.rand(3, self.height, self.width)
        return {"blur": blur, "sharp": sharp}


def test_metrics_with_main_network():
    print("=== Testing Metrics Compatibility with Main Network ===")

    # Load config and override device for CPU test
    config = load_config("config/default.yaml")
    device = "cpu"

    # Build main models
    _, _, restoration_net, physical_layer = build_models_from_config(config, device)

    # Create dummy val loader
    dataset = DummyBlurSharpDataset(length=2, height=64, width=64)
    val_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    evaluator = PerformanceEvaluator(device=device)

    # Speed up inference time measurement for test
    evaluator._measure_inference_time = lambda model, device, input_shape=(1, 3, 64, 64), warmup=1, repeat=2: (
        PerformanceEvaluator._measure_inference_time(model, device, input_shape, warmup, repeat)
    )

    metrics = evaluator.evaluate(
        restoration_net=restoration_net,
        physical_layer=physical_layer,
        val_loader=val_loader,
        device=device,
        smoothness_grid_size=8
    )

    expected_keys = {
        "PSNR", "SSIM", "LPIPS", "Reblur_MSE", "PSF_Smoothness",
        "Params(M)", "FLOPs(GMACs)", "Inference(ms)"
    }

    missing = expected_keys - set(metrics.keys())
    assert not missing, f"Missing metrics: {missing}"

    # Basic sanity checks
    assert torch.isfinite(torch.tensor(metrics["PSNR"])), "PSNR should be finite"
    assert torch.isfinite(torch.tensor(metrics["SSIM"])), "SSIM should be finite"
    assert torch.isfinite(torch.tensor(metrics["Reblur_MSE"])), "Reblur_MSE should be finite"
    assert torch.isfinite(torch.tensor(metrics["PSF_Smoothness"])), "PSF_Smoothness should be finite"

    print("✅ Metrics compatibility test passed!")


if __name__ == "__main__":
    test_metrics_with_main_network()
