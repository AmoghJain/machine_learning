import os
import glob
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class ImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        """
        root_A: Path to the folder containing Domain A images (Monet)
        root_B: Path to the folder containing Domain B images (Photos)
        transforms_: List of PyTorch transforms to apply
        """
        self.transform = transforms.Compose(transform)
        
        # Get all image paths from the folders
        # We use glob to find .jpg, .png, etc.
        self.files_A = sorted(glob.glob(os.path.join(root_A, '*.*')))
        self.files_B = sorted(glob.glob(os.path.join(root_B, '*.*')))
        
        # Safety check
        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise RuntimeError(f"No images found in {root_A} or {root_B}!")

    def __getitem__(self, index):
        # 1. Get Image A (Monet)
        # We use modulo % because index might exceed len(files_A)
        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert('RGB'))

        # 2. Get Image B (Photo)
        # CycleGAN is UNPAIRED. We pick a random photo to pair with the Monet.
        # This prevents the model from memorizing fixed pairs.
        index_B = random.randint(0, len(self.files_B) - 1)
        item_B = self.transform(Image.open(self.files_B[index_B]).convert('RGB'))

        return {'monet': item_A, 'image': item_B}

    def __len__(self):
        # The epoch length is determined by the LARGER dataset.
        return max(len(self.files_A), len(self.files_B))