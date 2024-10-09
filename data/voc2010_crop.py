import os
from typing import Literal

import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

object_categories = ["bird", "cat", "cow", "dog", "horse", "sheep"]


class CUBDataset(Dataset):

    def __init__(self, root, data_name, my_type, train=True, transform=None, loader=default_loader, is_frac=None,
                 sample_num=-1):
        self.root = os.path.expanduser(root)
        self.data_name = data_name
        self.my_type = my_type
        self.transform = transform
        self.loader = loader
        self.train = train
        self.is_frac = is_frac
        self.sample_num = sample_num
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted.")

    def _load_metadata(self):
        data_txt = None
        if self.data_name in object_categories:
            data_txt = "%s_info.txt" % self.data_name
        elif self.data_name == "cub":
            if self.my_type == "ori":
                data_txt = "image_info.txt"
            else:
                data_txt = "cubsample_info.txt"
        elif self.data_name == "helen":
            data_txt = "helen_info.txt"
        elif self.data_name == "voc_multi":
            data_txt = "animal_info.txt"

        self.data = pd.read_csv(os.path.join(self.root, data_txt),
                                names=["img_id", "file_path", "target", "is_training_img"])
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        if self.is_frac is not None:
            self.data = self.data[self.data.target == self.is_frac]

        if self.sample_num != -1:
            self.data = self.data[0:self.sample_num]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except FileNotFoundError:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, row.file_path)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, sample.file_path)
        target = sample.target  # Targets start at 1 by default, so shift to 0
        img = self.loader(str(path))

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_dataset(dataset_dir, dataset_name, split: Literal["train", "test"] = None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    label = None if split == "train" else 0
    if split == "train":
        train_set = CUBDataset(dataset_dir, dataset_name, "iccnn", train=True, transform=transform, is_frac=label)
        return [train_set, ]
    elif split == "test":
        test_set = CUBDataset(dataset_dir, dataset_name, "iccnn", train=False, transform=transform, is_frac=label)
        return [test_set, ]
    else:
        train_set = CUBDataset(dataset_dir, dataset_name, "iccnn", train=True, transform=transform, is_frac=label)
        test_set = CUBDataset(dataset_dir, dataset_name, "iccnn", train=False, transform=transform, is_frac=label)
        return [train_set, test_set]
