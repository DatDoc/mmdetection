import torch
from torch.utils.data import DataLoader, Dataset
import os
import glob
import cv2 

class PSPNetDataset(Dataset):
    def __init__(self, image_root, transforms=None):
        self.image_root = image_root
        self.transforms = transforms

    def __len__(self):
        return len(os.listdir(self.image_root))
    
    def __getitem__(self, idx):
        # image_paths = os.path.join(self.image_root, row.image_id)
        image_paths = glob.glob(os.path.join(self.image_root, '*.png'))

        # preprocess image
        img = cv2.imread(image_paths[idx], 1)
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        
        return image_paths[idx], img
        
