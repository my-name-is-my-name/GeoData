"""
Тестирование обученной модели на тестовой выборке
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from pathlib import Path

from models.unet import get_unet_model
from utils.dataset import SatelliteDataset, get_validation_augmentation
from utils.metrics import SegmentationMetrics, AreaCalculator
from utils.visualization import save_prediction_visualization, visualize_batch_predictions

import matplotlib.pyplot as plt


def test_model(model, test_loader, device, threshold=0.5, save_dir='./test_results'):
    """
    Тестирование модели на тестовой выборкеcSxu!wNBnJ4Bjtd
    
    Args:
        model: обученная модель
        test_loader: DataLoader с тестовыми данными
        device: устройство (cpu/cuda)
        threshold: порог для бинаризации
        save_dir: директория для сохранения результатов
    """
    model.eval()
    
    # Создаём директорию для результатов
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'visualizations'), exist_ok=True)
    
    # Метрики
    all_metrics = {
        'iou': [],
        'dice': [],
        'pixel_accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Результаты для каждого изображения
    detailed_results = []
    
    print('Starting evaluation...')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']
            
            # Предсказания
            outputs = model(images)
            predictions = torch.sigmoid(outputs.squeeze(1))
            
            # Вычисляем метрики для батча
            batch_metrics = SegmentationMetrics.calculate_all_metrics(
                predictions, masks, threshold
            )
            
            # Сохраняем метрики
            for key in all_metrics:
                all_metrics[key].append(batch_metrics[key])
            
            # Обрабатываем каждое изображение в батче
            for i in range(len(images)):
                img = images[i]
                mask = masks[i]
                pred = predictions[i]
                filename = filenames[i]
                
                # Вычисляем площадь
                area_m2 = AreaCalculator.calculate_building_area(
                    pred.cpu().numpy(),
                    pixel_size_m=0.3,
                    threshold=threshold
                )
                
                coverage = AreaCalculator.calculate_coverage_percentage(
                    pred.cpu().numpy(),
                    threshold=threshold
                )
                
                # Сохраняем детальную информацию
                result = {
                    'filename': filename,
                    'iou': SegmentationMetrics.iou_score(pred.unsqueeze(0), mask.unsqueeze(0), threshold),
                    'dice': SegmentationMetrics.dice_score(pred.unsqueeze(0), mask.unsqueeze(0), threshold),
                    'area_m2': area_m2,
                    'coverage_percent': coverage
                }
                detailed_results.append(result)
                
                # Сохраняем визуализацию (только для первых 50 примеров)
                if batch_idx * test_loader.batch_size + i < 50:
                    viz_path = os.path.join(
                        save_dir, 
                        'visualizations', 
                        f'{Path(filename).stem}_result.png'
                    )
                    save_prediction_visualization(
                        img, mask, pred, 
                        viz_path, 
                        threshold
                    )
    
    # Усредняем метрики
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    # Выводим результаты
    print('\n' + '=' * 60)
    print('TEST RESULTS')
    print('=' * 60)
    for key, value in avg_metrics.items():
        print(f'{key.upper()}: {value:.4f}')
    print('=' * 60)
    
    # Сохраняем метрики в JSON
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    # Сохраняем детальные результаты в CSV
    df = pd.DataFrame(detailed_results)
    df.to_csv(os.path.join(save_dir, 'detailed_results.csv'), index=False)
    
    # Статистика по площади
    print('\nArea Statistics:')
    print(f'  Mean area: {df["area_m2"].mean():.2f} m²')
    print(f'  Median area: {df["area_m2"].median():.2f} m²')
    print(f'  Min area: {df["area_m2"].min():.2f} m²')
    print(f'  Max area: {df["area_m2"].max():.2f} m²')
    
    # Создаём графики
    create_test_visualizations(df, all_metrics, save_dir)
    
    return avg_metrics, detailed_results


def create_test_visualizations(df, all_metrics, save_dir):
    """
    Создаёт визуализации результатов тестирования
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # IoU distribution
    axes[0, 0].hist(df['iou'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('IoU Score Distribution')
    axes[0, 0].axvline(df['iou'].mean(), color='red', linestyle='--', label=f'Mean: {df["iou"].mean():.3f}')
    axes[0, 0].legend()
    
    # Dice distribution
    axes[0, 1].hist(df['dice'], bins=30, edgecolor='black')
    axes[0, 1].set_xlabel('Dice Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Dice Score Distribution')
    axes[0, 1].axvline(df['dice'].mean(), color='red', linestyle='--', label=f'Mean: {df["dice"].mean():.3f}')
    axes[0, 1].legend()
    
    # Area distribution
    axes[0, 2].hist(df['area_m2'], bins=30, edgecolor='black')
    axes[0, 2].set_xlabel('Area (m²)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Building Area Distribution')
    axes[0, 2].axvline(df['area_m2'].mean(), color='red', linestyle='--', label=f'Mean: {df["area_m2"].mean():.0f}')
    axes[0, 2].legend()
    
    # Coverage distribution
    axes[1, 0].hist(df['coverage_percent'], bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Coverage (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Building Coverage Distribution')
    axes[1, 0].axvline(df['coverage_percent'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df["coverage_percent"].mean():.2f}%')
    axes[1, 0].legend()
    
    # Box plots for metrics
    metrics_data = [all_metrics['iou'], all_metrics['dice'], all_metrics['f1']]
    axes[1, 1].boxplot(metrics_data, labels=['IoU', 'Dice', 'F1'])
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Metrics Box Plot')
    axes[1, 1].grid(True)
    
    # Scatter: IoU vs Area
    axes[1, 2].scatter(df['area_m2'], df['iou'], alpha=0.5)
    axes[1, 2].set_xlabel('Area (m²)')
    axes[1, 2].set_ylabel('IoU Score')
    axes[1, 2].set_title('IoU vs Building Area')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_analysis.png'), dpi=150)
    plt.close()
    
    print(f'\n✓ Visualizations saved to {save_dir}/test_analysis.png')


def main(args):
    """
    Главная функция
    """
    # Устройство
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Загрузка модели
    print(f'Loading model from {args.checkpoint_path}...')
    model = get_unet_model(
        model_type=args.model_type,
        n_channels=3,
        n_classes=1
    )
    
    checkpoint = torch.load(args.checkpoint_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f'✓ Model loaded (epoch {checkpoint.get("epoch", "unknown")})')
    print(f'  Best Val IoU: {checkpoint.get("val_iou", "unknown")}')
    
    # Создаём тестовый датасет
    print('Creating test dataset...')
    test_dataset = SatelliteDataset(
        args.test_images_dir,
        args.test_masks_dir if args.test_masks_dir else None,
        transform=get_validation_augmentation(args.image_size),
        mode='test'
    )
    
    print(f'Test samples: {len(test_dataset)}')
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Тестирование
    avg_metrics, detailed_results = test_model(
        model,
        test_loader,
        device,
        threshold=args.threshold,
        save_dir=args.save_dir
    )
    
    print(f'\n✓ Results saved to {args.save_dir}/')
    print('  - test_metrics.json: average metrics')
    print('  - detailed_results.csv: per-image results')
    print('  - test_analysis.png: visualizations')
    print('  - visualizations/: individual predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test segmentation model')
    
    # Пути
    parser.add_argument('--test_images_dir', type=str, required=True,
                        help='Path to test images')
    parser.add_argument('--test_masks_dir', type=str, default=None,
                        help='Path to test masks (optional)')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='Directory to save results')
    
    # Параметры модели
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet_bilinear', 'unet++'],
                        help='Model architecture')
    
    # Параметры тестирования
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binarization')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    main(args)
