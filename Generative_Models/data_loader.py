import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ICLEVRDataset(Dataset):
    def __init__(self, json_path, objects_path, image_dir, transform=None, is_train=True):
        # 加載標註資料與物體映射
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        with open(objects_path, 'r') as f:
            self.objects_map = json.load(f)

        self.image_dir = image_dir
        self.is_test = isinstance(self.annotations, list)  # 檢查是否為測試模式
        self.image_files = list(self.annotations.keys()) if not self.is_test else None
        self.num_classes = len(self.objects_map)

        # 僅在訓練模式下應用 transform
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]) if transform is None else transform
        else:
            self.transform = transform  # 測試模式下不應該有 transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 取得標籤
        if self.is_test:
            labels_list = self.annotations[idx]
        else:
            img_name = self.image_files[idx]
            labels_list = self.annotations[img_name]

        label_tensor = torch.zeros(self.num_classes)
        for label in labels_list:
            label_tensor[self.objects_map[label]] = 1.0

        if self.is_test:
            return label_tensor  # 測試模式返回標籤
        else:
            # 訓練模式：讀取圖像並應用轉換
            img_path = os.path.join(self.image_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            return self.transform(image) if self.transform else image, label_tensor
