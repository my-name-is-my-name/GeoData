"""
Streamlit приложение для анализа площади застройки
с поддержкой sliding window для больших изображений
"""
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import io
import os
import sys
from pathlib import Path
import tempfile
import math

# Конфигурация
st.set_page_config(
    page_title="Анализ застройки | Спутниковые снимки",
    page_icon="🏘️",
    layout="wide"
)


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def add_safe_globals():
    """Разрешаем безопасную загрузку numpy объектов"""
    try:
        import torch.serialization
        import numpy as np
        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    except:
        pass


@st.cache_resource
def load_trained_model(model_path, device='cpu'):
    """
    Загружает предобученную модель U-Net
    """
    add_safe_globals()

    try:
        sys.path.append('.')
        from models.unet import UNet

        model = UNet(n_channels=3, n_classes=1, bilinear=True)

        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        model_info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_iou': checkpoint.get('val_iou', 'unknown'),
            'type': 'U-Net with Bilinear Upsampling'
        }

        return model, model_info

    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None, None


def get_transform(img_size=512):
    """Трансформации для инференса"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def create_overlay(image, mask, alpha=0.6):
    """Создаёт наложение маски на изображение"""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    mask_binary = (mask > 0.5).astype(np.uint8)
    colored_mask[mask_binary == 1] = (255, 0, 0)  # Красный цвет

    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def extract_geotiff_metadata(filepath):
    """Пытается извлечь масштаб из метаданных GeoTIFF"""
    try:
        import rasterio

        with rasterio.open(filepath) as src:
            if src.crs:
                # Есть геопривязка
                pixel_size_x = src.transform[0]
                pixel_size_y = abs(src.transform[4])
                pixel_size = (pixel_size_x + pixel_size_y) / 2

                # Если в градусах - грубый перевод в метры
                if src.crs.is_geographic:
                    pixel_size *= 111000  # ~111 км в градусе
                    return pixel_size, "geographic_crs"
                else:
                    return pixel_size, "projected_crs"
    except ImportError:
        return None, "rasterio_not_installed"
    except Exception:
        return None, "no_geotiff_metadata"

    return None, "not_geotiff"


def predict_sliding_window(model, image, device, patch_size=512, overlap=128):
    """
    Sliding window предсказание для больших изображений
    """
    h, w = image.shape[:2]

    # Создаём пустую маску для всего изображения
    full_prediction = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    # Шаг с учётом перекрытия
    stride = patch_size - overlap

    # Считаем количество патчей
    num_patches_h = math.ceil((h - overlap) / stride)
    num_patches_w = math.ceil((w - overlap) / stride)
    total_patches = num_patches_h * num_patches_w

    if total_patches == 0:
        return full_prediction

    # Прогресс-бар
    progress_bar = st.progress(0, text=f"Обработка 0/{total_patches} патчей")

    # Создаём весовую маску для blending
    y_coords, x_coords = np.meshgrid(
        np.arange(patch_size),
        np.arange(patch_size),
        indexing='ij'
    )
    center = patch_size // 2
    distances = np.sqrt((y_coords - center) ** 2 + (x_coords - center) ** 2)
    patch_weights = np.clip(1 - distances / (patch_size / 2), 0, 1)

    transform = get_transform(patch_size)
    patch_counter = 0

    # Обрабатываем все патчи
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Определяем границы патча
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch_h = y_end - y
            patch_w = x_end - x

            # Пропускаем слишком маленькие патчи
            if patch_h < 64 or patch_w < 64:
                continue

            # Вырезаем патч
            patch = image[y:y_end, x:x_end]

            # Если патч меньше нужного размера - добавляем паддинг
            if patch_h < patch_size or patch_w < patch_size:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
                padded_patch[:patch_h, :patch_w] = patch
            else:
                padded_patch = patch

            # Трансформации и предсказание
            transformed = transform(image=padded_patch)
            input_tensor = transformed['image'].unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.sigmoid(output).squeeze().cpu().numpy()

            # Обрезаем предсказание до реального размера патча
            patch_pred = prediction[:patch_h, :patch_w]
            patch_weights_cropped = patch_weights[:patch_h, :patch_w]

            # Добавляем к полной маске с весами
            full_prediction[y:y_end, x:x_end] += patch_pred * patch_weights_cropped
            weight_map[y:y_end, x:x_end] += patch_weights_cropped

            # Обновляем прогресс
            patch_counter += 1
            progress = patch_counter / total_patches
            progress_bar.progress(
                progress,
                text=f"Обработка {patch_counter}/{total_patches} патчей ({progress:.1%})"
            )

    progress_bar.empty()

    # Нормализуем результат
    weight_map[weight_map == 0] = 1  # избегаем деления на 0
    full_prediction = full_prediction / weight_map

    return full_prediction


def smart_predict(model, image, device, patch_size=512):
    """
    Умное предсказание: выбирает стратегию в зависимости от размера
    """
    h, w = image.shape[:2]

    # Для больших изображений используем sliding window
    if h > 1024 or w > 1024:
        st.info(f"🔄 Большое изображение ({w}×{h}): использую sliding window")
        overlap = patch_size // 4  # 25% перекрытие
        return predict_sliding_window(model, image, device, patch_size, overlap)
    else:
        # Для маленьких изображений - простой ресайз
        st.info(f"⚡ Маленькое изображение ({w}×{h}): прямой анализ")
        transform = get_transform(patch_size)
        transformed = transform(image=image)
        input_tensor = transformed['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.sigmoid(output).squeeze().cpu().numpy()

        # Возвращаем к оригинальному размеру
        prediction = cv2.resize(prediction, (w, h))
        return prediction


def count_buildings_opencv(binary_mask, min_area=25):
    """
    Подсчёт количества зданий через OpenCV
    min_area: минимальная площадь в пикселях для учёта объекта
    """
    # Находим контуры
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Фильтруем слишком маленькие контуры
    valid_contours = []
    building_areas_px = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            valid_contours.append(contour)
            building_areas_px.append(area)

    return len(valid_contours), building_areas_px


# ==================== ОСНОВНОЙ ИНТЕРФЕЙС ====================

def main():
    st.title("🏘️ Анализ площади застройки по спутниковым снимкам")

    st.markdown("""
    **Внимание:** Модель оптимизирована для снимков с разрешением **0.3 м/пиксель** (Inria Aerial Dataset).  
    Для других снимков точность может снижаться.
    """)

    # ========== БОКОВАЯ ПАНЕЛЬ ==========
    with st.sidebar:
        st.header("⚙️ Настройки")

        # Только одна настройка для пользователя
        sensitivity = st.slider(
            "Чувствительность обнаружения",
            min_value=1,
            max_value=10,
            value=5,
            help="Выше значение - меньше найденных объектов"
        )
        threshold = sensitivity / 10

        # Техническая информация (скрытая)
        with st.expander("ℹ️ Технические детали"):
            st.caption("""
            - Модель: U-Net Bilinear
            - Обучена на: Inria Aerial Dataset
            - Оптимальное разрешение: 0.3 м/пиксель
            - IoU: ~66%
            - Обработка: sliding window для больших изображений
            """)

    # ========== ЗАГРУЗКА МОДЕЛИ ==========
    MODEL_PATH = "./weights/best_model.pth"

    if not os.path.exists(MODEL_PATH):
        st.error(f"""
        ⚠️ **Модель не найдена!**

        Путь: `{MODEL_PATH}`

        Чтобы обучить модель:
        ```bash
        python train_seg.py --train_images_dir ./data/train/images ...
        ```
        """)
        return

    if 'model' not in st.session_state:
        with st.spinner("Загрузка модели..."):
            model, model_info = load_trained_model(MODEL_PATH, 'cpu')
            if model:
                st.session_state['model'] = model
                st.success("✅ Модель загружена")
            else:
                st.error("Не удалось загрузить модель")
                return

    # ========== ЗАГРРУЗКА ИЗОБРАЖЕНИЯ ==========
    st.header("📤 Загрузите спутниковый снимок")

    uploaded_file = st.file_uploader(
        " ",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        label_visibility="collapsed",
        help="Поддерживаемые форматы: PNG, JPG, TIFF, GeoTIFF"
    )

    if uploaded_file:
        # Сохраняем временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        try:
            # Загружаем изображение для показа
            image = Image.open(temp_path)
            image_np = np.array(image.convert('RGB'))
            h, w = image_np.shape[:2]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Ваш снимок")
                st.image(image_np, use_container_width=True)
                st.caption(f"Размер: {w} × {h} пикселей")

            # ========== ОПРЕДЕЛЕНИЕ МАСШТАБА ==========
            st.subheader("📏 Укажите масштаб снимка")

            pixel_size = None
            scale_source = ""

            # Попытка 1: Из метаданных GeoTIFF
            if uploaded_file.name.lower().endswith(('.tif', '.tiff')):
                with st.spinner("Проверка метаданных GeoTIFF..."):
                    pixel_size, metadata_status = extract_geotiff_metadata(temp_path)

                    if pixel_size:
                        st.success(f"✅ Масштаб определён из метаданных: {pixel_size:.4f} м/пиксель")
                        scale_source = "geotiff_metadata"
                    else:
                        st.info("ℹ️ Не удалось определить масштаб из метаданных")

            # Если не определили из метаданных - предлагаем варианты
            if pixel_size is None:
                tab1, tab2, tab3 = st.tabs([
                    "📐 Метров на пиксель",
                    "📏 Размеры участка",
                    "ℹ️ Рекомендации"
                ])

                with tab1:
                    st.markdown("**Укажите разрешение снимка:**")

                    # Примеры типичных значений
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("0.3 м", use_container_width=True, key="btn_03"):
                            st.session_state['selected_pixel_size'] = 0.3
                            st.session_state['selected_scale_source'] = "inria_default"
                    with col_b:
                        if st.button("0.5 м", use_container_width=True, key="btn_05"):
                            st.session_state['selected_pixel_size'] = 0.5
                            st.session_state['selected_scale_source'] = "manual"
                    with col_c:
                        if st.button("1.0 м", use_container_width=True, key="btn_10"):
                            st.session_state['selected_pixel_size'] = 1.0
                            st.session_state['selected_scale_source'] = "manual"

                    # Ручной ввод
                    manual_size = st.number_input(
                        "Или введите своё значение:",
                        min_value=0.01,
                        max_value=100.0,
                        value=0.3,
                        step=0.01,
                        format="%.3f",
                        key="manual_input"
                    )

                    if st.button("Использовать это значение", key="use_manual"):
                        st.session_state['selected_pixel_size'] = manual_size
                        st.session_state['selected_scale_source'] = "manual"

                with tab2:
                    st.markdown("**Укажите реальные размеры участка:**")

                    col_x, col_y = st.columns(2)
                    with col_x:
                        width_m = st.number_input(
                            "Ширина участка (м)",
                            min_value=1.0,
                            max_value=100000.0,
                            value=100.0,
                            step=1.0,
                            key="width_input"
                        )
                    with col_y:
                        height_m = st.number_input(
                            "Высота участка (м)",
                            min_value=1.0,
                            max_value=100000.0,
                            value=100.0,
                            step=1.0,
                            key="height_input"
                        )

                    if width_m and height_m and w > 0 and h > 0:
                        pixel_size_x = width_m / w
                        pixel_size_y = height_m / h
                        pixel_size_avg = (pixel_size_x + pixel_size_y) / 2

                        st.info(f"Расчётный масштаб: {pixel_size_avg:.4f} м/пиксель")

                        if st.button("Использовать расчётный масштаб", key="use_calc"):
                            st.session_state['selected_pixel_size'] = pixel_size_avg
                            st.session_state['selected_scale_source'] = "calculated"

                with tab3:
                    st.markdown("**Рекомендации по масштабу:**")

                    st.write("""
                    **Оптимально для этой модели:** 0.3 м/пиксель

                    **Типичные значения:**
                    - Maxar/Planet: 0.3-0.5 м
                    - Airbus: 0.5-1.5 м  
                    - Sentinel-2: 10 м
                    - Дроны: 0.01-0.1 м

                    **Inria Aerial Dataset:** 0.3 м/пиксель
                    """)

                    if st.button("Использовать 0.3 м (Inria)", key="use_inria"):
                        st.session_state['selected_pixel_size'] = 0.3
                        st.session_state['selected_scale_source'] = "inria_default"

            # ========== АНАЛИЗ ==========
            # Всегда показываем кнопку анализа
            st.markdown("---")

            # Проверяем, выбран ли масштаб (из метаданных или из session_state)
            current_pixel_size = pixel_size
            current_scale_source = scale_source

            if current_pixel_size is None and 'selected_pixel_size' in st.session_state:
                current_pixel_size = st.session_state['selected_pixel_size']
                current_scale_source = st.session_state.get('selected_scale_source', 'manual')

            # Показываем текущий выбранный масштаб
            if current_pixel_size is not None:
                st.info(f"📏 Выбранный масштаб: **{current_pixel_size:.4f} м/пиксель**")

                # Предупреждение если масштаб не 0.3
                if abs(current_pixel_size - 0.3) > 0.05:  # Если отличается более чем на 5%
                    st.warning(f"""
                    ⚠️ **Внимание:** Вы указали масштаб {current_pixel_size:.3f} м/пиксель

                    Модель оптимизирована для **0.3 м/пиксель** (Inria Aerial Dataset).
                    Для этого разрешения точность может быть ниже.
                    """)

            # Кнопка запуска анализа
            if st.button("🚀 Начать анализ", type="primary", use_container_width=True):

                # Проверяем, что масштаб выбран
                if current_pixel_size is None:
                    st.error("❌ Сначала укажите масштаб снимка!")
                    st.stop()

                with st.spinner("Обработка изображения..."):
                    try:
                        # Предсказание с sliding window
                        model = st.session_state['model']

                        # Используем patch_size=512 как при обучении
                        prediction = smart_predict(model, image_np, 'cpu', patch_size=512)

                        # Бинаризация
                        binary_mask = (prediction > threshold).astype(np.uint8)

                        # Визуализация
                        overlay = create_overlay(image_np, binary_mask, alpha=0.6)

                        # Расчёт основной площади
                        building_pixels = np.sum(binary_mask)
                        area_m2 = building_pixels * (current_pixel_size ** 2)
                        coverage = (building_pixels / binary_mask.size) * 100

                        # Статистика по объектам
                        num_buildings, building_areas_px = count_buildings_opencv(binary_mask, min_area=25)

                        # Конвертируем площади в м²
                        building_areas_m2 = [area * (current_pixel_size ** 2) for area in building_areas_px]

                        # Показываем результаты
                        with col2:
                            st.subheader("Результаты")

                            # Вкладки
                            tab_viz, tab_stats = st.tabs(["Визуализация", "Статистика"])

                            with tab_viz:
                                st.image(overlay, use_container_width=True)
                                st.caption(f"Найдено зданий: {num_buildings}")

                            with tab_stats:
                                # Основные метрики
                                st.metric(
                                    "Площадь застройки",
                                    f"{area_m2:,.0f} м²",
                                    delta=None,
                                    help=f"При {current_pixel_size:.3f} м/пиксель"
                                )

                                st.metric(
                                    "Процент застройки",
                                    f"{coverage:.1f}%"
                                )

                                st.metric(
                                    "Количество зданий",
                                    f"{num_buildings}"
                                )

                                st.metric(
                                    "Пиксели зданий",
                                    f"{building_pixels:,}"
                                )

                                # Дополнительная статистика если есть здания
                                if num_buildings > 0:
                                    st.markdown("---")
                                    st.subheader("📊 Статистика по зданиям")

                                    col_stat1, col_stat2 = st.columns(2)
                                    with col_stat1:
                                        st.write(f"**Средняя площадь:** {np.mean(building_areas_m2):.0f} м²")
                                        st.write(f"**Медианная площадь:** {np.median(building_areas_m2):.0f} м²")
                                    with col_stat2:
                                        st.write(f"**Минимальная площадь:** {np.min(building_areas_m2):.0f} м²")
                                        st.write(f"**Максимальная площадь:** {np.max(building_areas_m2):.0f} м²")

                                # Информация о масштабе
                                st.info(f"""
                                **Источник масштаба:** {current_scale_source}
                                **Значение:** {current_pixel_size:.4f} м/пиксель
                                **Площадь пикселя:** {current_pixel_size ** 2:.6f} м²
                                """)

                        # Скачивание результатов
                        st.subheader("💾 Скачать результаты")

                        col_dl1, col_dl2 = st.columns(2)

                        with col_dl1:
                            # Маска
                            mask_pil = Image.fromarray((binary_mask * 255).astype(np.uint8))
                            mask_bytes = io.BytesIO()
                            mask_pil.save(mask_bytes, format='PNG')

                            st.download_button(
                                "📥 Маска зданий (PNG)",
                                data=mask_bytes.getvalue(),
                                file_name="building_mask.png",
                                mime="image/png"
                            )

                        with col_dl2:
                            # Отчёт
                            report = f"""АНАЛИЗ ПЛОЩАДИ ЗАСТРОЙКИ

Файл: {uploaded_file.name}
Дата: {st.session_state.get('analysis_time', 'N/A')}

РАЗМЕРЫ:
- Изображение: {w} × {h} пикселей
- Масштаб: {current_pixel_size:.4f} м/пиксель (источник: {current_scale_source})
- Общая площадь кадра: {(w * h * current_pixel_size ** 2):,.0f} м²

РЕЗУЛЬТАТЫ:
- Площадь застройки: {area_m2:,.0f} м²
- Процент застройки: {coverage:.1f}%
- Количество зданий: {num_buildings}
- Пиксели зданий: {building_pixels:,}

СТАТИСТИКА ПО ЗДАНИЯМ:
- Средняя площадь: {np.mean(building_areas_m2) if num_buildings > 0 else 0:.0f} м²
- Медианная площадь: {np.median(building_areas_m2) if num_buildings > 0 else 0:.0f} м²
- Минимальная площадь: {np.min(building_areas_m2) if num_buildings > 0 else 0:.0f} м²
- Максимальная площадь: {np.max(building_areas_m2) if num_buildings > 0 else 0:.0f} м²

ПАРАМЕТРЫ:
- Чувствительность: {sensitivity}/10
- Порог: {threshold:.2f}
- Модель: U-Net Bilinear
- Обработка: {'Sliding window' if h > 1024 or w > 1024 else 'Прямой анализ'}
"""

                            st.download_button(
                                "📥 Отчёт (TXT)",
                                data=report,
                                file_name="building_analysis.txt",
                                mime="text/plain"
                            )

                    except Exception as e:
                        st.error(f"❌ Ошибка обработки: {str(e)}")

        finally:
            # Удаляем временный файл
            try:
                os.unlink(temp_path)
            except:
                pass

    else:
        # Инструкция при загрузке
        st.info("👆 Загрузите спутниковый снимок для начала анализа")

        with st.expander("📋 Подробнее о системе"):
            st.markdown("""
            ### Как это работает:

            1. **Загружаете** спутниковый снимок
            2. **Указываете масштаб** одним из способов:
               - Автоматически из метаданных GeoTIFF
               - Вручную: "метров на пиксель"
               - Через реальные размеры участка
            3. **Получаете** площадь застройки в м²

            ### Особенности:
            - Оптимально для снимков из Inria Aerial Dataset (0.3 м/пикс)
            - Для других снимков точность может быть ниже
            - Для больших изображений используется **Sliding window** - снимок разрезается на патчи 512x512 пикселей, для патчей строятся предсказания, затем восстанавливается оригинальный размер
            - Для маленьких изображений используется паддинг

            ### Форматы:
            - PNG, JPG, JPEG (обычные изображения)
            - TIFF, GeoTIFF (с поддержкой метаданных)
            """)


if __name__ == '__main__':
    import datetime

    st.session_state['analysis_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Инициализируем session_state переменные если их нет
    if 'selected_pixel_size' not in st.session_state:
        st.session_state['selected_pixel_size'] = None
    if 'selected_scale_source' not in st.session_state:
        st.session_state['selected_scale_source'] = ""

    main()