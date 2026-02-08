"""
Метрики для оценки качества сегментации и детекции
"""
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


class SegmentationMetrics:
    """
    Метрики для семантической сегментации
    """
    
    @staticmethod
    def iou_score(predictions, targets, threshold=0.5, smooth=1e-6):
        """
        Intersection over Union (IoU) / Jaccard Index
        
        Args:
            predictions: tensor (B, H, W) или (B, 1, H, W)
            targets: tensor (B, H, W)
            threshold: порог для бинаризации предсказаний
        """
        predictions = (predictions > threshold).float()
        targets = targets.float()
        
        intersection = (predictions * targets).sum(dim=(1, 2))
        union = predictions.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return iou.mean().item()
    
    @staticmethod
    def dice_score(predictions, targets, threshold=0.5, smooth=1e-6):
        """
        Dice Score / F1 Score для сегментации
        """
        predictions = (predictions > threshold).float()
        targets = targets.float()
        
        intersection = (predictions * targets).sum(dim=(1, 2))
        dice = (2. * intersection + smooth) / (predictions.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) + smooth)
        
        return dice.mean().item()
    
    @staticmethod
    def pixel_accuracy(predictions, targets, threshold=0.5):
        """
        Pixel Accuracy - процент правильно классифицированных пикселей
        """
        predictions = (predictions > threshold).float()
        targets = targets.float()
        
        correct = (predictions == targets).float()
        accuracy = correct.mean().item()
        
        return accuracy
    
    @staticmethod
    def precision_recall(predictions, targets, threshold=0.5):
        """
        Precision и Recall для сегментации
        """
        predictions = (predictions > threshold).float().cpu().numpy().flatten()
        targets = targets.float().cpu().numpy().flatten()
        
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)
        
        return precision, recall, f1
    
    @staticmethod
    def calculate_all_metrics(predictions, targets, threshold=0.5):
        """
        Вычисляет все метрики одновременно
        """
        iou = SegmentationMetrics.iou_score(predictions, targets, threshold)
        dice = SegmentationMetrics.dice_score(predictions, targets, threshold)
        accuracy = SegmentationMetrics.pixel_accuracy(predictions, targets, threshold)
        precision, recall, f1 = SegmentationMetrics.precision_recall(predictions, targets, threshold)
        
        return {
            'iou': iou,
            'dice': dice,
            'pixel_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class AreaCalculator:
    """
    Расчёт площади застройки в м²
    """
    
    @staticmethod
    def calculate_building_area(mask, pixel_size_m=0.3, threshold=0.5):
        """
        Вычисляет площадь зданий на маске
        
        Args:
            mask: binary mask (H, W) или tensor
            pixel_size_m: размер одного пикселя в метрах (зависит от разрешения снимка)
            threshold: порог для бинаризации
        
        Returns:
            area_m2: площадь в квадратных метрах
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Бинаризация
        binary_mask = (mask > threshold).astype(np.uint8)
        
        # Подсчёт пикселей зданий
        building_pixels = np.sum(binary_mask)
        
        # Площадь одного пикселя в м²
        pixel_area_m2 = pixel_size_m ** 2
        
        # Общая площадь
        area_m2 = building_pixels * pixel_area_m2
        
        return area_m2
    
    @staticmethod
    def calculate_coverage_percentage(mask, threshold=0.5):
        """
        Вычисляет процент застройки на изображении
        """
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        binary_mask = (mask > threshold).astype(np.uint8)
        total_pixels = mask.size
        building_pixels = np.sum(binary_mask)
        
        coverage = (building_pixels / total_pixels) * 100
        
        return coverage


class DetectionMetrics:
    """
    Метрики для детекции объектов (для альтернативной задачи)
    """
    
    @staticmethod
    def calculate_iou_boxes(box1, box2):
        """
        IoU для bounding boxes
        box format: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou
    
    @staticmethod
    def calculate_map(pred_boxes, pred_scores, true_boxes, iou_threshold=0.5):
        """
        Mean Average Precision (mAP)
        Упрощённая версия для одного класса
        """
        # Сортируем предсказания по уверенности
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        
        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched_gt = set()
        
        for i, pred_box in enumerate(pred_boxes):
            max_iou = 0
            max_idx = -1
            
            for j, true_box in enumerate(true_boxes):
                if j in matched_gt:
                    continue
                
                iou = DetectionMetrics.calculate_iou_boxes(pred_box, true_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(max_idx)
            else:
                fp[i] = 1
        
        # Cumulative sum
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision and Recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recall = tp_cumsum / (len(true_boxes) + 1e-6)
        
        # Average Precision
        ap = np.trapz(precision, recall)
        
        return ap


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Оценка модели на валидационном/тестовом датасете
    
    Args:
        model: обученная модель
        dataloader: DataLoader с данными
        device: устройство (cpu/cuda)
        threshold: порог для бинаризации
    
    Returns:
        dict: словарь с метриками
    """
    model.eval()
    
    all_metrics = {
        'iou': [],
        'dice': [],
        'pixel_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Предсказания
            outputs = model(images)
            predictions = torch.sigmoid(outputs)
            
            # Вычисляем метрики для батча
            metrics = SegmentationMetrics.calculate_all_metrics(predictions, masks, threshold)
            
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
    
    # Усредняем метрики
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_metrics
