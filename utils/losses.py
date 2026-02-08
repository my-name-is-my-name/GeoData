"""
Комбинированные функции потерь для сегментации
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss для сегментации
    Хорошо работает с несбалансированными классами
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        # predictions: (B, 1, H, W) или (B, H, W)
        # targets: (B, H, W)
        
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss - фокусируется на сложных примерах
    Полезен для несбалансированных классов
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, predictions, targets):
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class IoULoss(nn.Module):
    """
    IoU Loss (Intersection over Union)
    Напрямую оптимизирует метрику IoU
    """
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        total = (predictions + targets).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class CombinedLoss(nn.Module):
    """
    Комбинированная функция потерь: Dice + BCE
    
    Обоснование выбора:
    1. Dice Loss - хорошо работает с несбалансированными классами (здания vs фон)
       и напрямую оптимизирует метрику Dice Score
    2. Binary Cross Entropy - стабилизирует обучение и помогает на ранних этапах
    3. Комбинация даёт лучшие результаты, чем каждый loss по отдельности
    
    Args:
        dice_weight: вес для Dice Loss
        bce_weight: вес для BCE Loss
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    """
    Tversky Loss - обобщение Dice Loss
    Позволяет контролировать баланс между false positives и false negatives
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (predictions * targets).sum()
        FP = ((1 - targets) * predictions).sum()
        FN = (targets * (1 - predictions)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class CombinedLossV2(nn.Module):
    """
    Альтернативная комбинация: Dice + Focal
    
    Обоснование:
    1. Dice Loss - оптимизирует overlap между предсказанием и целью
    2. Focal Loss - фокусируется на сложных примерах, где модель не уверена
    3. Вместе они дают хороший баланс между точностью границ и уверенностью предсказаний
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLossV2, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)
        
        return self.dice_weight * dice + self.focal_weight * focal


class CombinedLossV3(nn.Module):
    """
    Тройная комбинация: Dice + BCE + IoU
    
    Обоснование:
    1. Dice - хорош для несбалансированных классов
    2. BCE - стабильность обучения
    3. IoU - напрямую оптимизирует основную метрику
    """
    def __init__(self, dice_weight=0.4, bce_weight=0.3, iou_weight=0.3):
        super(CombinedLossV3, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()
        self.iou_loss = IoULoss()
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        iou = self.iou_loss(predictions, targets)
        
        return (self.dice_weight * dice + 
                self.bce_weight * bce + 
                self.iou_weight * iou)


def get_loss_function(loss_type='combined'):
    """
    Фабрика для создания функции потерь
    
    Args:
        loss_type: 'dice', 'bce', 'focal', 'iou', 'combined', 'combined_v2', 'combined_v3'
    """
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'focal':
        return FocalLoss()
    elif loss_type == 'iou':
        return IoULoss()
    elif loss_type == 'combined':
        return CombinedLoss()
    elif loss_type == 'combined_v2':
        return CombinedLossV2()
    elif loss_type == 'combined_v3':
        return CombinedLossV3()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
