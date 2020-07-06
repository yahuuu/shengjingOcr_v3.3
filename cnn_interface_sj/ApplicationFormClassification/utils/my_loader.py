#coding:utf-8

import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def data_loadermy(bgr_np, img_path):

    # print("-->",bgr_np, img_path)
    # print(bgr_np.shape)
    h, w, _ = bgr_np.shape
    bgr_np = bgr_np[0:h, 0:h, :]
    rgb_np = cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB)
    rgb_np = cv2.resize(rgb_np, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    # rgb_np = cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB)[np.newaxis, :]
    # input_tensor = torch.from_numpy(rgb_np.transpose((0, 3, 1, 2)))
    # input_tensor = torch.from_numpy(rgb_np.transpose((2, 0, 1)))
    input_tensor = torch.from_numpy(rgb_np.transpose((2, 1, 0)))
    input_tensor = input_tensor.float().div(255).unsqueeze(0)
    # print(input_tensor.shape)
    # print(input_tensor.shape)
    return [(input_tensor, (img_path,))]

def data_loadermy1(img_pil, img_pth):
    img_pil = img_pil.crop((0, 0, img_pil.size[1], img_pil.size[1]))
    transform1 = transforms.Compose([transforms.Resize((224, 224), interpolation=Image.LANCZOS), transforms.ToTensor()])
    input_tensor = transform1(img_pil).unsqueeze(0)
    return [(input_tensor, (img_pth,))]


if __name__ == "__main__":
    img1 = 'C:\\work\\gy_classification_win_py3\\python_and_proj\\Classification_server\\test_img\\101001\\98.jpg'
    print(data_loadermy(img1))