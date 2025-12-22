
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import config

class FaceMaskDataset(Dataset):
    def __init__(self, file_list, root_dir, transforms=None):
        self.file_list = file_list
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback
            return self.__getitem__((idx + 1) % len(self))

        if self.transforms:
            image = self.transforms(image)

        masked_image = image.clone()
        c, h, w = masked_image.shape
        
        # Synthetic Mask
        mask_y_start = int(h * 0.50)
        mask_y_end = int(h * 0.95)
        mask_x_start = int(w * 0.15)
        mask_x_end = int(w * 0.85)
        
        # For [-1, 1], black is -1.
        masked_image[:, mask_y_start:mask_y_end, mask_x_start:mask_x_end] = -1.0 

        return masked_image, image

def get_transforms():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ])
