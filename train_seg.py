"""
Обучение модели семантической сегментации с возможностью продолжения обучения
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import json

from models.unet import get_unet_model
from utils.dataset import SatelliteDataset, get_training_augmentation, get_validation_augmentation
from utils.losses import get_loss_function
from utils.metrics import SegmentationMetrics, evaluate_model
from utils.visualization import plot_training_history, visualize_batch_predictions

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_IO_ENABLE_TIFF_IGNORE_READ_WARNINGS'] = '1'


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 device, checkpoint_dir='./weights', log_dir='./logs', resume_from=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.resume_from = resume_from

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir)
        self.best_val_iou = 0.0
        self.start_epoch = 1

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_iou': [], 'val_iou': [],
            'train_dice': [], 'val_dice': [],
            'learning_rate': []
        }

        if self.resume_from:
            self.load_checkpoint(self.resume_from)

    def load_checkpoint(self, checkpoint_path):
        try:
            if os.path.exists(checkpoint_path):
                print(f"Загрузка checkpoint из: {checkpoint_path}")

                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                if 'history' in checkpoint:
                    self.history = checkpoint['history']

                if 'val_iou' in checkpoint:
                    self.best_val_iou = checkpoint['val_iou']

                self.start_epoch = checkpoint.get('epoch', 1) + 1

                print(f"Загружен checkpoint эпохи {self.start_epoch - 1}")
                print(f"Лучший Val IoU: {self.best_val_iou:.4f}")
                print(f"Продолжаем с эпохи: {self.start_epoch}")

                return True
            else:
                print(f"Файл checkpoint не найден: {checkpoint_path}")
                return False

        except Exception as e:
            print(f"Ошибка загрузки checkpoint: {e}")
            return False

    def train_epoch(self, epoch):
        self.model.train()

        running_loss = 0.0
        all_ious = []
        all_dices = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')

        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)

            loss = self.criterion(torch.sigmoid(outputs.squeeze(1)), masks)

            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                predictions = torch.sigmoid(outputs.squeeze(1))
                iou = SegmentationMetrics.iou_score(predictions, masks)
                dice = SegmentationMetrics.dice_score(predictions, masks)

            running_loss += loss.item()
            all_ious.append(iou)
            all_dices.append(dice)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{iou:.4f}',
                'dice': f'{dice:.4f}'
            })

        avg_loss = running_loss / len(self.train_loader)
        avg_iou = np.mean(all_ious)
        avg_dice = np.mean(all_dices)

        return avg_loss, avg_iou, avg_dice

    def validate(self, epoch):
        self.model.eval()

        running_loss = 0.0
        all_ious = []
        all_dices = []

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')

        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)

                outputs = self.model(images)
                predictions = torch.sigmoid(outputs.squeeze(1))

                loss = self.criterion(predictions, masks)

                iou = SegmentationMetrics.iou_score(predictions, masks)
                dice = SegmentationMetrics.dice_score(predictions, masks)

                running_loss += loss.item()
                all_ious.append(iou)
                all_dices.append(dice)

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'iou': f'{iou:.4f}',
                    'dice': f'{dice:.4f}'
                })

        avg_loss = running_loss / len(self.val_loader)
        avg_iou = np.mean(all_ious)
        avg_dice = np.mean(all_dices)

        return avg_loss, avg_iou, avg_dice

    def save_checkpoint(self, epoch, val_iou, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_iou': val_iou,
            'history': self.history,
            'best_val_iou': self.best_val_iou
        }

        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f'✓ Saved best model with IoU: {val_iou:.4f}')

    def train(self, num_epochs, scheduler=None):
        print(f'Starting training for {num_epochs} epochs...')
        print(f'Device: {self.device}')
        print(f'Train batches: {len(self.train_loader)}')
        print(f'Val batches: {len(self.val_loader)}')
        print(f'Start epoch: {self.start_epoch}')
        if self.resume_from:
            print(f'Resuming from: {self.resume_from}')
        print('-' * 60)

        for epoch in range(self.start_epoch, num_epochs + 1):
            train_loss, train_iou, train_dice = self.train_epoch(epoch)
            val_loss, val_iou, val_dice = self.validate(epoch)

            if scheduler is not None:
                scheduler.step(val_loss)
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else \
                self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_iou'].append(train_iou)
            self.history['val_iou'].append(val_iou)
            self.history['train_dice'].append(train_dice)
            self.history['val_dice'].append(val_dice)
            self.history['learning_rate'].append(current_lr)

            self.writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
            self.writer.add_scalars('IoU', {'train': train_iou, 'val': val_iou}, epoch)
            self.writer.add_scalars('Dice', {'train': train_dice, 'val': val_dice}, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)

            print(f'\nEpoch {epoch}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Train Dice: {train_dice:.4f}')
            print(f'  Val Loss:   {val_loss:.4f} | Val IoU:   {val_iou:.4f} | Val Dice:   {val_dice:.4f}')
            print(f'  LR: {current_lr:.6f}')

            is_best = val_iou > self.best_val_iou
            if is_best:
                self.best_val_iou = val_iou

            self.save_checkpoint(epoch, val_iou, is_best)

            with open(os.path.join(self.log_dir, 'history.json'), 'w') as f:
                json.dump(self.history, f, indent=4)

        print('\n' + '=' * 60)
        print(f'Training completed!')
        print(f'Best validation IoU: {self.best_val_iou:.4f}')
        print('=' * 60)

        self.writer.close()

        fig = plot_training_history(self.history)
        fig.savefig(os.path.join(self.log_dir, 'training_history.png'), dpi=150)
        plt.close(fig)


def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    if device.type == 'mps' and args.num_workers > 0:
        print(f'MPS: setting num_workers to 0')
        args.num_workers = 0

    print('Creating datasets...')
    train_dataset = SatelliteDataset(
        args.train_images_dir,
        args.train_masks_dir,
        transform=get_training_augmentation(args.image_size),
        mode='train'
    )

    val_dataset = SatelliteDataset(
        args.val_images_dir,
        args.val_masks_dir,
        transform=get_validation_augmentation(args.image_size),
        mode='val'
    )

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f'Creating model: {args.model_type}')
    model = get_unet_model(
        model_type=args.model_type,
        n_channels=3,
        n_classes=1
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    print(f'Using loss: {args.loss_type}')
    criterion = get_loss_function(args.loss_type)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # FIXED: убрал verbose параметр
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from
    )

    trainer.train(num_epochs=args.num_epochs, scheduler=scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train segmentation model')

    parser.add_argument('--train_images_dir', type=str, required=True)
    parser.add_argument('--train_masks_dir', type=str, required=True)
    parser.add_argument('--val_images_dir', type=str, required=True)
    parser.add_argument('--val_masks_dir', type=str, required=True)

    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet_bilinear', 'unet++'])
    parser.add_argument('--loss_type', type=str, default='dice',
                        choices=['dice', 'bce', 'focal', 'iou', 'combined', 'combined_v2', 'combined_v3'])

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--checkpoint_dir', type=str, default='./weights')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()

    main(args)