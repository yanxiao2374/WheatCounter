import os
import cv2
import numpy as np
import click


@click.command()
@click.option('--size', type=click.Choice(['1024', '2048']), required=True, help='裁剪尺寸: 1024 或 2048')
@click.option('--input-dir', type=str, required=True, help='输入图片目录路径')
@click.option('--output-dir', type=str, required=True, help='输出图片目录路径')
def cut_images(size, input_dir, output_dir):
    size = int(size)
    count = 1

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        img = cv2.imread(file_path)
        h, w = img.shape[:2]
        h_step = size
        w_step = size

        h_tiles = h // h_step
        w_tiles = w // w_step

        for y in range(0, h_tiles * h_step, h_step):
            for x in range(0, w_tiles * w_step, w_step):
                tile = img[y:y + h_step, x:x + w_step]
                tile = cv2.resize(tile, (1024, 1024))  # 保持输出大小为1024x1024
                save_name = os.path.join(output_dir, f"douth_{str(count).zfill(3)}.png")
                cv2.imwrite(save_name, tile)
                count += 1


if __name__ == '__main__':
    cut_images()