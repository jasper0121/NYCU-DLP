import os
import torch
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()  # read train/valid/test splits

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def load_dataset(data_path, mode, augmentation=False):
    if mode not in {"train", "valid", "test"}:
        raise ValueError("mode 必須是 'train', 'valid' 或 'test'")
    
    '''設定圖片與標註資料夾路徑，若資料不存在，則自動下載'''
    images_dir = os.path.join(data_path, "images")
    annotations_dir = os.path.join(data_path, "annotations")
    if not (os.path.exists(images_dir) and os.path.exists(annotations_dir)):
        OxfordPetDataset.download(data_path)
    
    '''載入基本資料集，若啟用資料增強且是訓練或驗證模式，則回傳增強後的資料集'''
    dataset = SimpleOxfordPetDataset(data_path, mode=mode)
    if augmentation and mode in {"train", "valid"}:
        return AugmentedDataset(dataset) # 包一個資料增強class再回傳
    else:
        return dataset

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        # 定義 8 種增強組合：4 種旋轉角度 (0, 90, 180, 270) × 是否水平翻轉 (True/False)
        self.aug = [(k, flip) for k in range(4) for flip in [False, True]]
    
    def __len__(self): # 資料集長度是 原始資料數 × 8 (每筆資料對應 8 種增強組合)
        return len(self.dataset) * len(self.aug)
    
    def __getitem__(self, idx):
        '''計算當前圖片index是對應哪個原始圖片，以及是哪種旋轉 / 是否鏡像'''
        orig_idx, aug_idx = divmod(idx, len(self.aug))
        k, flip = self.aug[aug_idx]

        '''取得原始圖片(包含 image, mask, trimap)'''
        sample = self.dataset[orig_idx]
        image, mask, trimap = sample["image"], sample["mask"], sample["trimap"]
        
        '''處理圖片，根據k和flip決定該圖片是否旋轉 / 鏡像翻轉'''
        if k: # k = 0, 1, 2, 3 -> 旋轉 (k*90 度)
            image, mask, trimap = (np.rot90(arr, k, axes=(1, 2)).copy() for arr in (image, mask, trimap))

        if flip: # 若需要水平翻轉，則沿寬度軸進行反轉
            image, mask, trimap = (arr[:, :, ::-1].copy() for arr in (image, mask, trimap))
        
        return {"image": image, "mask": mask, "trimap": trimap} # 回傳處理後的圖片
