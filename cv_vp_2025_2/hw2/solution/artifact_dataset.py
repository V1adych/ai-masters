import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

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
            shuffle=True,
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
