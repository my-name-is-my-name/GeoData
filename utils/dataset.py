"""
Кастомный Dataset для работы со спутниковыми снимками зданий
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SatelliteDataset(Dataset):
    """
    Кастомный Dataset для сегментации зданий на спутниковых снимках
    
    Args:
        images_dir: путь к директории с изображениями
        masks_dir: путь к директории с масками
        transform: трансформации для аугментации
        mode: 'train', 'val' или 'test'
    """
    
    def __init__(self, images_dir, masks_dir=None, transform=None, mode='train'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mode = mode
        
        # Получаем список файлов
        self.images = sorted([f for f in os.listdir(images_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        if masks_dir and os.path.exists(masks_dir):
            self.masks = sorted([f for f in os.listdir(masks_dir) 
                               if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        else:
            self.masks = None
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загружаем маску (если есть)
        if self.masks is not None:
            mask_path = os.path.join(self.masks_dir, self.masks[idx])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Бинаризация маски (0 - фон, 1 - здание)
            mask = (mask > 127).astype(np.float32)
        else:
            # Для тестовой выборки без масок
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Применяем аугментации
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        return {
            'image': image,
            'mask': mask,
            'filename': self.images[idx]
        }


class DetectionDataset(Dataset):
    """
    Dataset для детекции зданий (bounding boxes)
    """
    
    def __init__(self, images_dir, annotations_path=None, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        
        self.images = sorted([f for f in os.listdir(images_dir) 
                            if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        # Загружаем аннотации (если есть)
        self.annotations = {}
        if annotations_path and os.path.exists(annotations_path):
            # Здесь нужно загрузить аннотации в формате COCO или аналогичном
            pass
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Получаем bounding boxes (заглушка)
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target


def get_training_augmentation(img_size=512):
    """
    Аугментации для обучения
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.HueSaturationValue(p=1.0),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_validation_augmentation(img_size=512):
    """
    Аугментации для валидации/теста (только resize и normalize)
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def create_dataloaders(train_images_dir, train_masks_dir, 
                       val_images_dir, val_masks_dir,
                       batch_size=8, num_workers=4, img_size=512):
    """
    Создаёт DataLoader'ы для обучения и валидации
    """
    train_dataset = SatelliteDataset(
        train_images_dir, 
        train_masks_dir,
        transform=get_training_augmentation(img_size),
        mode='train'
    )
    
    val_dataset = SatelliteDataset(
        val_images_dir,
        val_masks_dir,
        transform=get_validation_augmentation(img_size),
        mode='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
