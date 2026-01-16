from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import os

class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        self.files = sorted([
            f for f in os.listdir(lr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        lr_path = os.path.join(self.lr_dir, fname)
        lr_img = self.to_tensor(Image.open(lr_path).convert("RGB"))

        if self.hr_dir is not None:
            hr_path = os.path.join(self.hr_dir, fname)
            hr_img = self.to_tensor(Image.open(hr_path).convert("RGB"))
            return lr_img, hr_img, fname

        return lr_img, fname
