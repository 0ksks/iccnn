import os
import pandas as pd
from torchvision import datasets, transforms
from typing import Literal


# 创建一个自定义数据集类
class CUBDataset(datasets.ImageFolder):
    def __init__(self, dataset_dir, data, transform=None, train=True):
        self.root = dataset_dir
        self.dataset_dir = dataset_dir
        self.data = data[data["is_training"] == (1 if train else 0)]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.dataset_dir, "images", self.data.iloc[idx]["image_path"])
        label = self.data.iloc[idx]["class_id"] - 1  # 类别ID从0开始
        image = datasets.folder.default_loader(str(image_path))  # 加载图像

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataset(dataset_dir, split: Literal["train", "test"] = None):
    dataset_dir = os.path.join(dataset_dir, "CUB_200_2011")

    # 解析图片、标签和训练/测试集划分
    def parse_data(dataset_dir):
        images_path = os.path.join(dataset_dir, "images.txt")
        labels_path = os.path.join(dataset_dir, "image_class_labels.txt")
        train_test_split_path = os.path.join(dataset_dir, "train_test_split.txt")

        # 读取文件
        images = pd.read_csv(images_path, sep=" ", header=None, names=["image_id", "image_path"])
        labels = pd.read_csv(labels_path, sep=" ", header=None, names=["image_id", "class_id"])
        train_test_split = pd.read_csv(train_test_split_path, sep=" ", header=None, names=["image_id", "is_training"])

        # 合并数据
        parsed_data = pd.merge(images, labels, on="image_id")
        parsed_data = pd.merge(parsed_data, train_test_split, on="image_id")

        return parsed_data

    data = parse_data(dataset_dir)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像缩放到 224x224
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 使用 ImageNet 的均值和标准差进行标准化
    ])
    if split == "train":
        train_dataset = CUBDataset(dataset_dir, data, transform=transform, train=True)
        return [train_dataset, ]
    elif split == "test":
        test_dataset = CUBDataset(dataset_dir, data, transform=transform, train=False)
        return [test_dataset, ]
    else:
        train_dataset = CUBDataset(dataset_dir, data, transform=transform, train=True)
        test_dataset = CUBDataset(dataset_dir, data, transform=transform, train=False)
        return [train_dataset, test_dataset]


def single_category_dataset(dataset: CUBDataset):
    single_category_data = dataset.data[dataset.data["class_id"] <= 2]
    dataset.data = single_category_data
    return dataset
