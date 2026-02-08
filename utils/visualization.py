"""
Визуализация результатов сегментации
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from matplotlib.colors import ListedColormap


def visualize_sample(image, mask=None, prediction=None, alpha=0.4):
    """
    Визуализация одного примера
    
    Args:
        image: исходное изображение (C, H, W) tensor или (H, W, C) numpy
        mask: ground truth маска (H, W)
        prediction: предсказанная маска (H, W)
        alpha: прозрачность маски
    """
    # Конвертируем tensor в numpy если нужно
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            image = np.transpose(image, (1, 2, 0))
    
    # Денормализация изображения
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Определяем количество subplot'ов
    num_plots = 1 + (mask is not None) + (prediction is not None)
    
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    # Исходное изображение
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    current_idx = 1
    
    # Ground truth маска
    if mask is not None:
        axes[current_idx].imshow(image)
        axes[current_idx].imshow(mask, alpha=alpha, cmap='Reds')
        axes[current_idx].set_title('Ground Truth')
        axes[current_idx].axis('off')
        current_idx += 1
    
    # Предсказание
    if prediction is not None:
        axes[current_idx].imshow(image)
        axes[current_idx].imshow(prediction, alpha=alpha, cmap='Blues')
        axes[current_idx].set_title('Prediction')
        axes[current_idx].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_comparison(image, mask, prediction, threshold=0.5):
    """
    Сравнение ground truth и предсказания
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Бинаризация предсказания
    pred_binary = (prediction > threshold).astype(np.uint8)
    mask_binary = mask.astype(np.uint8)
    
    # Создаём RGB overlay
    # Зелёный: True Positive
    # Красный: False Positive
    # Синий: False Negative
    overlay = np.zeros((*mask.shape, 3))
    
    tp = (pred_binary == 1) & (mask_binary == 1)  # True Positive
    fp = (pred_binary == 1) & (mask_binary == 0)  # False Positive
    fn = (pred_binary == 0) & (mask_binary == 1)  # False Negative
    
    overlay[tp] = [0, 1, 0]  # Зелёный
    overlay[fp] = [1, 0, 0]  # Красный
    overlay[fn] = [0, 0, 1]  # Синий
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    axes[3].imshow(image)
    axes[3].imshow(overlay, alpha=0.5)
    axes[3].set_title('Overlay (Green=TP, Red=FP, Blue=FN)')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(history):
    """
    График обучения модели
    
    Args:
        history: dict с ключами 'train_loss', 'val_loss', 'train_iou', 'val_iou' и т.д.
    """
    epochs = len(history.get('train_loss', []))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    axes[0].plot(history.get('train_loss', []), label='Train Loss', marker='o')
    axes[0].plot(history.get('val_loss', []), label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # IoU
    axes[1].plot(history.get('train_iou', []), label='Train IoU', marker='o')
    axes[1].plot(history.get('val_iou', []), label='Val IoU', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU Score')
    axes[1].set_title('Training and Validation IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig


def visualize_batch_predictions(images, masks, predictions, num_samples=4, threshold=0.5):
    """
    Визуализация нескольких примеров из батча
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        mask = masks[i].cpu().numpy()
        prediction = predictions[i].cpu().numpy()
        
        # Денормализация
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Original
        axes[i, 0].imshow(image)
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # Ground Truth
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(mask, alpha=0.5, cmap='Reds')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(image)
        axes[i, 2].imshow(prediction > threshold, alpha=0.5, cmap='Blues')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    return fig


def create_overlay_image(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    Создаёт наложение маски на изображение для сохранения
    
    Args:
        image: numpy array (H, W, 3)
        mask: numpy array (H, W)
        alpha: прозрачность
        color: цвет маски в RGB
    
    Returns:
        overlay: numpy array (H, W, 3)
    """
    # Убедимся, что изображение в правильном диапазоне
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Создаём цветную маску
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0.5] = color
    
    # Наложение
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return overlay


def save_prediction_visualization(image, mask, prediction, save_path, threshold=0.5):
    """
    Сохраняет визуализацию предсказания
    """
    fig = visualize_comparison(image, mask, prediction, threshold)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
