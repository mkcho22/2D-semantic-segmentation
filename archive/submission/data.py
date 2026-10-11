import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class CarlaDataset(Dataset):
    def __init__(self, root_dir, split, image_transform, label_transform):
        self.split_dir = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.split_dir, "RGB")
        self.label_dir = os.path.join(self.split_dir, "Label")
        
        self.images = sorted(os.listdir(self.img_dir))
        self.labels = sorted(os.listdir(self.label_dir))
        
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        label_name = self.labels[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        image = self.image_transform(image)

        label_np_temp = np.array(label, dtype=np.uint8)
        if label_np_temp.ndim == 3:
            label_pil = Image.fromarray(label_np_temp[:, :, 0])
        else:
            label_pil = label
        label_pil = self.label_transform(label_pil)
        label_np = np.array(label_pil, dtype=np.int64)
        label_np[label_np == 255] = 0
        label = torch.from_numpy(label_np)

        return image, label

def setup_data_loaders(BATCH_SIZE, DATA_DIR, INPUT_HEIGHT, INPUT_WIDTH,NUM_CLASSES):
    image_transform = transforms.Compose([
        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    label_transform = transforms.Compose([
        transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
    ])

    train_dataset = CarlaDataset(root_dir=DATA_DIR, split="Training", 
                             image_transform=image_transform, 
                             label_transform=label_transform)
    val_dataset = CarlaDataset(root_dir=DATA_DIR, split="Validation", 
                           image_transform=image_transform, 
                           label_transform=label_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,pin_memory=False)

    print(f"Training data: {len(train_dataset)} images")
    print(f"Validation data: {len(val_dataset)} images")
    print("--- Data loading complete ---")

    return train_loader, val_loader