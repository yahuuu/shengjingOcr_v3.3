# -*- coding:utf-8 -*-
import glob
import os

from PIL import Image
import torch.utils.data as data
import torchvision


class ImageFolder():
    def __call__(self, root_path, transform):
        obtained_dataset = torchvision.datasets.ImageFolder(root_path,
                                                            transform=transform)
        return  obtained_dataset


class ApplicationForm(data.Dataset):
    def __init__(self, root_path, val=False,
                 is_test=False, transform=None,
                 classes_index=None):
        self.val = val
        self.test = is_test
        self.transform = transform
        self.root_path = root_path
        self.classes_index = classes_index

        if self.test:
            self.test_images = self.get_images_path(root_path)
        else:
            self.images = self.get_images_path(root_path)

    def __getitem__(self, index):
        if self.test:
            img_path = self.test_images[index]
            label = str(img_path)
        else:
            img_path = self.images[index]

            # The full path has the needed label.
            # print("classes_index","-"*20, self.classes_index) # 4 test
            # print("--"*20, img_path) # 4 test
            label = self.classes_index[img_path.split('/')[-2]]

        data = Image.open(img_path)
        # print("1*"*10, data)
        # Crop the left part of the whole image, because the difference is here.
        data = data.crop((0, 0, data.size[1], data.size[1]))
        # print("2*"*10, data)

        if self.transform:
            data = self.transform(data)
        # print("3*"*10, data, label)
        return data, label

    def __len__(self):
        if self.test:
            return len(self.test_images)
        else:
            return len(self.images)

    @staticmethod
    def get_images_path(root_path):
        if os.path.isdir(root_path): # 收到的是图片路径 如/tmp/img
            all_images_path_ls = list(glob.glob(root_path + '/*/*.jpg'))
            all_images_path_ls += list(glob.glob(root_path + '/*/*.JPG'))
            all_images_path_ls += list(glob.glob(root_path + '/*/*.JPEG'))
            all_images_path_ls += list(glob.glob(root_path + '/*/*.jpeg'))
        else: # 收到的是图片的绝对路径的list , /tmp/1.jpg
            all_images_path_ls = [root_path]
        return all_images_path_ls
