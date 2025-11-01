# src/dataset.py
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, random
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

class CustomCelebAPairs(Dataset):
    def __init__(self, root="data/celeba/img_align_celeba", transform=transform):
        self.root = root
        self.files = sorted([f for f in os.listdir(root) if f.endswith(".jpg")])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cover_path = os.path.join(self.root, self.files[idx])
        cover = Image.open(cover_path).convert("RGB")
        secret_path = os.path.join(self.root, random.choice(self.files))
        secret = Image.open(secret_path).convert("RGB")
        return self.transform(cover), self.transform(secret)

def get_dataloader(batch_size=2, num_workers=0):
    dataset = CustomCelebAPairs()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
