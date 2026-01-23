import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DPDDDataset(Dataset):
    """
    DPDD (Dual-Pixel Defocus Deblurring) Dataset.
    Expects the following directory structure:
    root_dir/
        train/ (or val/ or test/)
            blur/
                img1.png
                ...
            sharp/
                img1.png
                ...
    """

    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., ./data/dpdd_1024).
            mode (str): One of 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode

        # Ensure transform is callable
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

        # Determine sub-directories
        self.split_dir = os.path.join(root_dir, mode)
        self.blur_dir = os.path.join(self.split_dir, 'blur')
        self.sharp_dir = os.path.join(self.split_dir, 'sharp')

        if not os.path.exists(self.blur_dir) or not os.path.exists(self.sharp_dir):
            raise FileNotFoundError(f"Blur or Sharp directory not found in {self.split_dir}")

        # Get file lists
        # We assume filenames are identical in blur/ and sharp/ folders after preprocessing
        self.blur_files = sorted([f for f in os.listdir(self.blur_dir) if self._is_image(f)])
        self.sharp_files = sorted([f for f in os.listdir(self.sharp_dir) if self._is_image(f)])

        # Verify integrity
        assert len(self.blur_files) == len(self.sharp_files), \
            f"Mismatch number of images in {self.blur_dir} and {self.sharp_dir}"

    def _is_image(self, filename):
        return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        # Retrieve filenames
        blur_filename = self.blur_files[idx]
        sharp_filename = self.sharp_files[idx]
        
        # In strict mode, we'd assert they are the same name. 
        # But if we rely on sorted order (like in the preprocessing script), 
        # we assume idx corresponds to the same scene.
        # Given the preprocess script ensures corresponding files, we proceed.

        blur_path = os.path.join(self.blur_dir, blur_filename)
        sharp_path = os.path.join(self.sharp_dir, sharp_filename)

        # Open images and convert to RGB
        blur_img = Image.open(blur_path).convert('RGB')
        sharp_img = Image.open(sharp_path).convert('RGB')

        # Apply transforms
        # If a custom transform is passed (e.g. for augmentation), it should handle both images or 
        # we apply it carefully. 
        # Typically for deblurring, geometric transforms (flip/crop) must be identical for both.
        # Photometric transforms might differ.
        
        # For this implementation, we assume `self.transform` handles the tuple (blur, sharp) 
        # OR we just apply ToTensor if no transform is given.
        
        # Apply transforms
        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)

        return blur_tensor, sharp_tensor
