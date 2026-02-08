import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import argparse


def create_patches(images_dir, masks_dir, output_dir, patch_size=256, stride=128):
    """
    Создает патчи из больших изображений
    """
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.tif', '.jpg', '.png'))])

    patch_info = []
    patch_count = 0

    for img_name in tqdm(image_files, desc="Creating patches"):
        # Загружаем изображение и маску
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, img_name)

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Warning: Could not load {img_name}")
            continue

        h, w = image.shape[:2]

        # Создаем патчи
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Вырезаем патч
                img_patch = image[y:y + patch_size, x:x + patch_size]
                mask_patch = mask[y:y + patch_size, x:x + patch_size]

                # Сохраняем
                patch_filename = f"{os.path.splitext(img_name)[0]}_{y}_{x}.png"

                cv2.imwrite(os.path.join(output_dir, 'images', patch_filename), img_patch)
                cv2.imwrite(os.path.join(output_dir, 'masks', patch_filename), mask_patch)

                # Сохраняем информацию о патче
                patch_info.append({
                    'patch_id': patch_filename,
                    'original_image': img_name,
                    'y': int(y),
                    'x': int(x),
                    'patch_size': patch_size,
                    'stride': stride
                })

                patch_count += 1

    # Сохраняем метаданные
    with open(os.path.join(output_dir, 'patch_info.json'), 'w') as f:
        json.dump(patch_info, f, indent=2)

    print(f"Created {patch_count} patches from {len(image_files)} images")
    print(f"Saved to: {output_dir}")

    return patch_count


def main():
    parser = argparse.ArgumentParser(description='Create patches from large images')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory with large images')
    parser.add_argument('--masks_dir', type=str, required=True, help='Directory with large masks')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for patches')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size')
    parser.add_argument('--stride', type=int, default=128, help='Stride (overlap = patch_size - stride)')

    args = parser.parse_args()

    create_patches(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        stride=args.stride
    )


if __name__ == '__main__':
    main()