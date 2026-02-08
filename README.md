# Satellite Building Segmentation

Проект по сегментации зданий на спутниковых снимках с использованием глубокого обучения.

## Описание проекта

Этот проект реализует систему автоматической сегментации зданий на спутниковых снимках с расчётом площади застройки. Используется архитектура U-Net для семантической сегментации и комбинированная функция потерь (Dice Loss + Binary Cross Entropy) для достижения лучших результатов.

## Структура репозитория

```
├── models/                     # Архитектуры моделей
│   ├── unet.py                 # U-Net для сегментации
├── utils/                      # Утилиты
│   ├── dataset.py              # Кастомный Dataset и DataLoader
│   ├── losses.py               # Функции потерь
│   ├── metrics.py              # Метрики оценки
│   └── visualization.py        # Визуализация результатов
├── weights/                    # Сохранённые веса моделей
├── logs/                       # Логи TensorBoard
├── train_seg.py                # Скрипт обучения сегментации
├── app.py                      # Streamlit приложение
├── requirements.txt            # Зависимости
└── README.md                   # Документация
```

## Установка и запуск

### Клонирование репозитория

```bash
git clone https://github.com/my-name-is-my-name/GeoData.git
cd GeoData
```

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Подготовка данных

Скачайте датасет и разместите его в следующей структуре:

```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### Обучение модели

```bash
python train_seg.py \
    --train_images_dir ./data/train/images \
    --train_masks_dir ./data/train/masks \
    --val_images_dir ./data/val/images \
    --val_masks_dir ./data/val/masks \
    --model_type unet_bilinear \
    --loss_type dice \
    --batch_size 16 \
    --num_epochs 20 \
    --image_size 256 \
    --learning_rate 1.5e-5
```

### Тестирование

```bash
python test_model.py \                                                                         
    --test_images_dir ./data/test/images \  
    --test_masks_dir ./data/test/masks \  
    --checkpoint_path ./weights/best_model.pth \
    --model_type unet_bilinear \
    --batch_size 16 \
    --image_size 256 \
    --threshold 0.5 \
```

**Параметры обучения:**

- `--model_type`: тип модели (`unet`, `unet++`, `unet_bilinear`)
- `--loss_type`: функция потерь (`dice`, `bce`, `focal`, `iou`, `combined`, `combined_v2`, `combined_v3`)
- `--batch_size`: размер батча
- `--num_epochs`: количество эпох
- `--learning_rate`: скорость обучения
- `--image_size`: размер входного изображения (по умолчанию 512)
- `--resume_from` : начать с сохраненного чекпоинта (./weights/last_checkpoint.pth)

### Запуск веб-приложения

```bash
streamlit run app.py
```

Приложение будет доступно по адресу: `http://localhost:8501`

По результатам обучения лучшая модель unet_bilinear (так как ее удалось обучить на бОльшем количестве эпох)
app.py работает именно с моделью unet_bilinear, для того чтобы запустить приложение с другой моделью нужно будет скорректировать код

Как это работает:

            1. Загружаете спутниковый снимок
            2. Указываете масштаб одним из способов:
               - Автоматически из метаданных GeoTIFF
               - Вручную: "метров на пиксель"
               - Через реальные размеры участка
            3. Получаете площадь застройки в м²

Особенности:

            - Оптимально для снимков из Inria Aerial Dataset (0.3 м/пикс)
            - Для других снимков точность может быть ниже
            - Для больших изображений используется Sliding window - 
            снимок разрезается на патчи 512x512 пикселей, 
            для патчей строятся предсказания, затем восстанавливается оригинальный размер
            - Для маленьких изображеений используется паддинг с сохранением изображения в центре

## Исследования и результаты

Все эксперименты и анализ результатов описаны в Jupyter notebook
https://colab.research.google.com/drive/1l_hgUgh9VuI1ZRwsVu0AP4_Tv5kXbpuH?usp=sharing





