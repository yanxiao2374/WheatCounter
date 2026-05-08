from __future__ import division
from models2 import base_patch16_384_gap
from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn
import os



def predict_count(img):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = base_patch16_384_gap(pretrained=False)
    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()
    pretrained_path="/data1/mazc/mrf/code/TransCrowd_Wheathead/model/full_stage_gap/checkpoint.pth"
    if os.path.isfile(pretrained_path):
        print("=> loading checkpoint '{}'".format(pretrained_path))
        checkpoint = torch.load(pretrained_path,map_location="cuda:0")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(pretrained_path))

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
        model(img)

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
    img = Image.open("/data1/mazc/mrf/Datasets/TransCrowd/heading/resize_images/train/heading_106_01.png").convert('RGB')
    predict_count(img)