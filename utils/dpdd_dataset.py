import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class DPDDDataset(Dataset):
    """
    DPDD (Dual-Pixel Defocus Deblurring) Dataset with synchronized random cropping.
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

    def __init__(self, root_dir, mode='train', crop_size=1024, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset (e.g., ./data/dpdd_1024).
            mode (str): One of 'train', 'val', 'test'.
            crop_size (int): Size of the random crop (default: 1024).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.crop_size = crop_size

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

        # Get original image dimensions (PIL uses (width, height))
        W_orig, H_orig = blur_img.size
        
        # Synchronized Random Crop:
        # Ensure both blur and sharp images are cropped at the EXACT same location
        if self.mode == 'train':
            # For training, perform random crop
            if H_orig >= self.crop_size and W_orig >= self.crop_size:
                # Generate random crop parameters once using PIL compatible method
                max_top = H_orig - self.crop_size
                max_left = W_orig - self.crop_size
                top = random.randint(0, max_top)
                left = random.randint(0, max_left)
                
                # PIL crop: (left, top, right, bottom)
                box = (left, top, left + self.crop_size, top + self.crop_size)
                blur_img = blur_img.crop(box)
                sharp_img = sharp_img.crop(box)
                crop_h, crop_w = self.crop_size, self.crop_size
            else:
                # If image is smaller than crop_size, pad it
                # (This shouldn't happen with DPDD at reasonable scale)
                if H_orig < self.crop_size or W_orig < self.crop_size:
                    pad_w = max(0, self.crop_size - W_orig)
                    pad_h = max(0, self.crop_size - H_orig)
                    # PIL pad: (left, top, right, bottom)
                    blur_img = Image.new('RGB', (W_orig + pad_w, H_orig + pad_h))
                    sharp_img = Image.new('RGB', (W_orig + pad_w, H_orig + pad_h))
                    blur_img.paste(Image.open(blur_path).convert('RGB'), (0, 0))
                    sharp_img.paste(Image.open(sharp_path).convert('RGB'), (0, 0))
                    top, left = 0, 0
                else:
                    top, left = 0, 0
                crop_h, crop_w = blur_img.height, blur_img.width
        else:
            # For validation/test, use center crop or full image
            if H_orig >= self.crop_size and W_orig >= self.crop_size:
                top = (H_orig - self.crop_size) // 2
                left = (W_orig - self.crop_size) // 2
                crop_h, crop_w = self.crop_size, self.crop_size
                
                # PIL crop
                box = (left, top, left + crop_w, top + crop_h)
                blur_img = blur_img.crop(box)
                sharp_img = sharp_img.crop(box)
            else:
                # For very small images, use as-is
                top, left = 0, 0
                crop_h, crop_w = blur_img.height, blur_img.width
        
        # Compute normalized crop_info: [top/H_orig, left/W_orig, crop_h/H_orig, crop_w/W_orig]
        # This represents the crop location in the original image coordinates
        crop_info = torch.tensor(
            [top / H_orig, left / W_orig, crop_h / H_orig, crop_w / W_orig],
            dtype=torch.float32
        )
        
        # Apply transforms to convert to tensors
        blur_tensor = self.transform(blur_img)
        sharp_tensor = self.transform(sharp_img)

        return {
            'blur': blur_tensor,
            'sharp': sharp_tensor,
            'crop_info': crop_info  # Global coordinate alignment info
        }
