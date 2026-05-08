from __future__ import division
import numpy as np
from models import base_patch16_384_token, base_patch16_384_gap
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import os
import cv2
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def fore_generater(img,save_path):
    img=preprocess_image(img)
    img = img.cuda()
    if len(img.shape) == 5:
        img = img.squeeze(0)
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():
        overlay = model(img)
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def preprocess_image(img):
    base=384
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    H, W = img.shape[:2]
    # 找到最接近 base 的倍数
    new_h = int(round(H / base) * base)
    new_w = int(round(W / base) * base)

    # 避免出现0
    new_h = max(new_h, base)
    new_w = max(new_w, base)
    # 缩放图像
    img = cv2.resize(img, (new_w, new_h))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img =transform(img) # [ C, H, W]
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
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = base_patch16_384_gap(pretrained=False)
    model = model.cuda()
    pretrained_path = "/data1/mazc/mrf/code/TransCrowd_Wheathead/model/gwhd_fore_sparse/model_best.pth"
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

    model.eval()
    img_dir = "/data1/mazc/mrf/Datasets/TransCrowd/GlobalWheat2020/resize_images/train"
    save_dir="/data1/mazc/mrf/Datasets/TransCrowd/GlobalWheat2020/foreground/train_sparse"

    # 支持的图片格式
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(img_extensions):
            img_path = os.path.join(img_dir, filename)
            img = Image.open(img_path).convert('RGB')
            save_path = os.path.join(save_dir, filename)
            fore_generater(img,save_path)