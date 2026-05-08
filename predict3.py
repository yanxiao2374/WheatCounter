from __future__ import division
import cv2
from models3 import base_patch16_384_gap
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import os



def predict_count(img_path_list):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    model = base_patch16_384_gap(pretrained=False)
    model = model.cuda()
    pretrained_path="/data1/mazc/mrf/code/TransCrowd_Wheathead/model/heading_fore_sparse_bc2/checkpoint.pth"
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

        model_parts = pretrained_path.split('/')
        model_name = model_parts[len(model_parts) - 2]
        img_parts = img_path.split('/')
        img_name = img_parts[len(img_parts) - 1]
        for i, overlay in enumerate(overlays):
            save_dir = f"./output/{model_name}"
            os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建

            save_path = os.path.join(save_dir, f"{img_name}_{i}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def preprocess_image(img):
    if img.width < 384 or img.height < 384:
        img = img.resize((max(384, img.width), max(384, img.height)), Image.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = transform(img)#[C, H, W]
    width, height = img.shape[2], img.shape[1]
    m = int(width / 384)
    n = int(height / 384)
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
        "/data1/mazc/mrf/Datasets/TransCrowd/heading/resize_images/train/heading_101_01.png",
        # "/data1/mazc/mrf/Datasets/TransCrowd/flowering/resize_images/train/flowering_101_01.png",
        # "/data1/mazc/mrf/Datasets/TransCrowd/filling/resize_images/train/filling_101_01.png",
        # "/data1/mazc/mrf/Datasets/TransCrowd/dough/resize_images/train/dough_101_01.png",
        # "/data1/mazc/mrf/Datasets/TransCrowd/heading/resize_images/train/heading_101_10.png",
        # "/data1/mazc/mrf/Datasets/TransCrowd/flowering/resize_images/train/flowering_101_10.png",
        # "/data1/mazc/mrf/Datasets/TransCrowd/filling/resize_images/train/filling_101_10.png",
        # "/data1/mazc/mrf/Datasets/TransCrowd/dough/resize_images/train/dough_101_10.png"
        ]
    fore_img_path_list = [
        "/data1/mazc/mrf/Datasets/TransCrowd/heading/foreground/train/heading_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/flowering/foreground/train/flowering_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/filling/foreground/train/filling_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_01.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/heading/foreground/train/heading_101_10.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/flowering/foreground/train/flowering_101_10.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/filling/foreground/train/filling_101_10.png",
        "/data1/mazc/mrf/Datasets/TransCrowd/dough/foreground/train/dough_101_10.png"
        ]
    # img_path="/data1/mazc/mrf/code/WheatCounter/fore_png/heading_101_01.png"
    predict_count(img_path_list)