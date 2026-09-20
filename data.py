import os
import tarfile
import urllib.request
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

URLS = [
    ("https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz", "imagenette2-320"),
]


def download_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for url, name in URLS:
        target = root / name / "train"
        if target.exists() and any(target.iterdir()):
            print(f"Using cached {target}")
            return target
        tgz = root / f"{name}.tgz"
        if not tgz.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, tgz)
        print(f"Extracting {tgz}")
        with tarfile.open(tgz, "r:gz") as tf:
            tf.extractall(root)
        if target.exists():
            return target
    raise RuntimeError("download failed")


def scan_folder(root):
    root = Path(root)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for cls in classes:
        for p in sorted((root / cls).iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                paths.append(str(p))
                labels.append(cls_to_idx[cls])
    return paths, labels, len(classes)


class MultiViewTransform:
    def __init__(self, gt, lt, ng, nl):
        self.gt = gt
        self.lt = lt
        self.ng = ng
        self.nl = nl

    def __call__(self, img):
        views = [self.gt(img) for _ in range(self.ng)] + [self.lt(img) for _ in range(self.nl)]
        return torch.stack(views)


class ImageDataset(Dataset):
    def __init__(self, root, tf):
        self.paths, self.labels, self.num_classes = scan_folder(root)
        self.tf = tf
        print(f"Found {len(self.paths)} images, {self.num_classes} classes in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


def make_train_transforms(size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean, std)
    g = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        transforms.RandomSolarize(threshold=0.5, p=0.1),
        transforms.ToTensor(),
        normalize,
    ])
    l = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        normalize,
    ])
    return g, l


def make_probe_transform(size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224), antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])