import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset


class MedicalFewShotDataset(Dataset):
    def __init__(self, image_dir, mask_dir, dataset_type="isic", support_size=3, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type.lower()
        self.support_size = support_size
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_dir) if not f.startswith('.')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith('.')])

    def __len__(self):
        return len(self.image_files)

    def _load_item(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        if self.dataset_type == "isic":
            mask_name = img_name.replace(".jpg", "_segmentation.png")
            mask_path = os.path.join(self.mask_dir, mask_name)
        elif self.dataset_type == "kvasir":
            mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        else:
            raise ValueError("Not supported dataset type!")

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()

        return image, mask, img_name

    def __getitem__(self, idx):
        query_img, query_mask, query_name = self._load_item(idx)

        available_indices = list(range(len(self.image_files)))
        available_indices.remove(idx)

        actual_support_size = min(self.support_size, len(available_indices))
        support_indices = random.sample(available_indices, actual_support_size)

        support_imgs, support_masks = [], []
        for s_idx in support_indices:
            s_img, s_mask, _ = self._load_item(s_idx)
            support_imgs.append(s_img)
            support_masks.append(s_mask)

        return torch.stack(support_imgs), torch.stack(support_masks), query_img, query_mask, query_name
