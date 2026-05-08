from __future__ import division
import cv2
from models3 import base_patch16_384_gap
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import os
import numpy as np



def predict_count(img_path_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = base_patch16_384_gap(pretrained=False)
    model = model.cuda()
    pretrained_path="/data1/mazc/mrf/code/TransCrowd_Wheathead/model/dough_distill/model_best.pth"
    if os.path.isfile(pretrained_path):
        print("=> loading checkpoint '{}'".format(pretrained_path))
        checkpoint = torch.load(pretrained_path,map_location="cuda:0")
        state_dict = checkpoint['state_dict']
        # 检查是否由 DataParallel 保存（参数名以 'module.' 开头）
        if any(k.startswith('module.') for k in state_dict.keys()):
            print("=> Detected DataParallel checkpoint, adapting keys...")
            # 去掉 'module.' 前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '', 1)
                new_state_dict[new_k] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(pretrained_path))

    for img_path in img_path_list:
        img = Image.open(img_path).convert('RGB')
        model.eval()
        img = preprocess_image(img)
        img = img.cuda()
        # print(img.shape)
        if len(img.shape) == 5:
            img = img.squeeze(0)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        # print("***", img.shape)
        with torch.no_grad():
            overlays = model(img)
        pic=reconstruct_image(overlays)
        model_parts = pretrained_path.split('/')
        model_name = model_parts[len(model_parts) - 2]
        img_parts = img_path.split('/')
        img_name = img_parts[len(img_parts) - 1]

        save_dir = f"./output/{model_name}"
        os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

        save_path = os.path.join(save_dir, f"{img_name}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(pic, cv2.COLOR_RGB2BGR))

def reconstruct_image(overlays, original_size=(1024, 1024)):
    """
     输入:
         overlays: list of 6 numpy images [384,384,3]
         original_size: tuple (H_original, W_original)
     输出:
         overlay_reconstructed: numpy image, 大小为 original_size
     """
    # -----------------------------
    # Step 1: 假设 6 张 patch 是 2×3 或 3×2
    # -----------------------------
    # 这里先固定为 2 rows × 3 cols
    n_rows, n_cols = 2, 3
    patch_h, patch_w = 384, 384

    # 拼接行
    rows = []
    for i in range(n_cols):
        row_patches = overlays[i * n_rows:(i + 1) * n_rows]
        row_concat = np.concatenate(row_patches, axis=0)  # 横向拼接
        rows.append(row_concat)

    # 拼接列
    overlay_big = np.concatenate(rows, axis=1)  # 纵向拼接，得到拼接后的大图

    # -----------------------------
    # Step 2: resize 回原始图像大小
    # -----------------------------
    W_orig,H_orig = original_size
    overlay_resized = cv2.resize(overlay_big, (W_orig, H_orig))

    return overlay_resized

def preprocess_image(img):
    """
    输入: PIL.Image
    输出: Tensor [6, 3, 384, 384]
    """
    # -----------------------------
    # Step 1: resize 到固定 1152×768 或 768×1152
    # -----------------------------
    w, h = img.size  # PIL.Image 的 width 和 height

    if w >= h:
        rate_1 = 1152.0 / w
        rate_2 = 768.0 / h
        img = cv2.resize(np.array(img), (0, 0), fx=rate_1, fy=rate_2)
    elif h > w:
        rate_1 = 1152.0 / h
        rate_2 = 768.0 / w
        img = cv2.resize(np.array(img), (0, 0), fx=rate_2, fy=rate_1)

    # -----------------------------
    # Step 2: ToTensor + Normalize
    # -----------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = transform(img)  # [C, H, W]

    # -----------------------------
    # Step 3: 384 × 384 不重叠裁剪
    # -----------------------------
    width, height = img.shape[2], img.shape[1]
    m = int(width / 384)
    n = int(height / 384)

    img_return = []

    for i in range(0, m):
        for j in range(0, n):

            if i == 0 and j == 0:
                img_return = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
            else:
                crop_img = img[:, j * 384: 384 * (j + 1), i * 384:(i + 1) * 384].cuda().unsqueeze(0)
                img_return = torch.cat([img_return, crop_img], 0).cuda()
    return img_return

if __name__=="__main__":
    img_path_list = [
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/images/train/dough_101.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/images/train/dough_102.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/images/train/dough_103.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/images/train/dough_104.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/images/train/dough_105.png",
        ]
    fore_img_path_list = [
        "/data1/mazc/mrf/Datasets/TransCrowd/heading/foreground/train/heading_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/heading/foreground/train/heading_101_10.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_10.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_10.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_10.png"
        ]
    # img_path="/data1/mazc/mrf/code/WheatCounter/fore_png/heading_101_01.png"
    predict_count(img_path_list)