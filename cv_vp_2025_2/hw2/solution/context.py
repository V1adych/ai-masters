import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
from sklearn.svm import SVC
import pandas as pd
import pickle
import sklearn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageDraw
from torchvision import transforms
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

import abc
from sklearn.metrics import roc_auc_score, mean_squared_error

from scipy.interpolate import interp1d
import numpy as np

from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
ssim = SSIM(reduction=None, return_full_image=False, return_contrast_sensitivity=False)
from lpips import LPIPS
from erqa import ERQA

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_image(image: np.array, device):
    image = image.astype(np.float64) / 255 * 2 - 1
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).float().to(device)

def metric(im1, im2, metric_fn):
    return metric_fn(preprocess_image(im1, device)[None, ...], preprocess_image(im2, device)[None, ...]).item()


def prepare_contour(bi: np.ndarray, sr: np.ndarray, mask: np.ndarray):
    assert bi.shape == sr.shape

    bi = cv2.resize(bi, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    bi = cv2.resize(bi, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST_EXACT)

    bi = bi.astype(np.float32)
    bi /= 255.0
    sr = sr.astype(np.float32)
    sr /= 255.0

    # Draw a border using a morphological gradient.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, [3] * 2)
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)

    shade = np.where(mask[..., None], [1.0, 1.0, 1.0], [0.5, 0.4, 0.4])

    bi *= shade
    bi = np.where(border[..., None], np.array([0.0, 0.0, 1.0])[None, None, :], bi)
    sr *= shade
    sr = np.where(border[..., None], np.array([0.0, 0.0, 1.0])[None, None, :], sr)

    bi *= 255.0
    sr *= 255.0
    bi = bi.astype(np.uint8)
    sr = sr.astype(np.uint8)

    bi = np.vstack((np.full((75, bi.shape[1], 3), 255, np.uint8), bi))
    sr = np.vstack((np.full((75, sr.shape[1], 3), 255, np.uint8), sr))

    bi = Image.fromarray(bi)
    draw = ImageDraw.Draw(bi)
    bi = np.asarray(bi)

    sr = Image.fromarray(sr)
    draw = ImageDraw.Draw(sr)
    sr = np.asarray(sr)

    return bi, sr


def interpolate(signal, scale_factor, kernel):
    original_length = len(signal)
    new_length = int(original_length * scale_factor)
    new_indices = np.linspace(0, original_length - 1, new_length)
    original_x = np.arange(original_length)
    distances = np.abs(new_indices[:, np.newaxis] - original_x[np.newaxis, :])
    weights = kernel(distances)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_sum[weights_sum == 0] = 1
    weights_normalized = weights / weights_sum
    interpolated = np.dot(weights_normalized, signal)
    return interpolated


def l2_norm(image1, image2):
    if image1.max() > 1:
        image1 = image1 / 255.0
    if image2.max() > 1:
        image2 = image2 / 255.0
    return np.linalg.norm(image1 - image2)


class ArtifactDataset(Dataset):
    def __init__(self,
                 img_dir: str,
                 labels_path: str):
        self.img_dir = img_dir
        self.labels_path = labels_path
        self.labels = pd.read_csv(self.labels_path, sep=',')

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx: int):
        # Каждому тестовому SR изображению соответствует маска артефакта, оригинальное изображение
        # и boolean метка, указывающая наличие/отсутствие артефакта
        # В случае False маска заполняется нулями
        sr_fn = self.labels.loc[idx, 'sr_fn']
        mask_fn = self.labels.loc[idx, 'mask_fn']
        gt_fn = self.labels.loc[idx, 'gt_fn']
        has_artifact = bool(self.labels.loc[idx, 'has_artifact'])

        img_path = os.path.join(self.img_dir, self.labels.loc[idx, 'folder'])

        img =  Image.open(f'{img_path}/{sr_fn}').convert('RGB')
        img_mask =  Image.open(f'{img_path}/{mask_fn}').convert('L')
        img_gt =  Image.open(f'{img_path}/{gt_fn}').convert('RGB')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Маска не нормализуется, т.к. уже содержит только 0 и 1
        transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])

        img_mask = transform_mask(img_mask)
        if not has_artifact:
            img_mask = torch.zeros_like(img_mask, dtype=torch.float32)

        return {'img': transform(img),
                'mask': img_mask,
                'gt': transform(img_gt),
                'has_artifact': has_artifact}
    

def create_dataloader(img_dir: str,
                      labels_path: str,
                      batch_size: int = 2,
                      num_workers: int = 0,
                      val_size: float = 0.2,
                      random_state: int = None):
    full_dataset = ArtifactDataset(img_dir, labels_path)

    seed = None
    if random_state is not None:
        seed = torch.Generator().manual_seed(random_state)
    train_dataset, val_dataset = random_split(full_dataset, [1 - val_size, val_size], generator=seed)
    
    train_loader, val_loader = None, None

    if len(train_dataset) > 0:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn
        )

    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader

# Агрегируем данные более удобным образом
# так, чтобы ключи словаря были верхней ступенью иерархии
def collate_fn(original_batch):
    new_batch = {}
    
    for el in original_batch:
        for key in el.keys():
            if key in new_batch:
                new_batch[key].append(el[key])
            else:
                new_batch[key] = [el[key]]

    for key in new_batch:
        if key != 'has_artifact':
            new_batch[key] = torch.stack(new_batch[key])
        else:
            new_batch[key] = torch.tensor(new_batch[key], dtype=torch.bool)

    return new_batch


def iou(preds, gt):
    dim = None
    if len(preds.shape) == 4:
        dim = (1, 2, 3)

    intersection = (preds * gt).sum(dim=dim)
    union = (preds + gt).clamp(max=1).sum(dim=dim)

    iou = intersection / (union + 1e-7)

    gt_has_art = (gt.sum(dim=dim) != 0)
    if dim is None and not gt_has_art.item():
        return 1
    elif dim is not None:
        for i in range(iou.shape[0]):
            if not gt_has_art[i]:
                iou[i] = 1

    return iou.mean()
