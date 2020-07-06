# -*- coding:utf-8 -*-
import torch
import torchvision
from PIL import Image

from data import dataset


class GenerateDataLoader(object):
    def __init__(self, opt):
        super(GenerateDataLoader, self).__init__()
        self.opt = opt


    def test_generation(self, img_abs_dir):
        test_transforms_list = [
            torchvision.transforms.Resize((self.opt.image_size_width, self.opt.image_size_height),
                                          interpolation=Image.LANCZOS),
            torchvision.transforms.ToTensor()
        ]
        test_loader = self.generate_dataloader(root_path= img_abs_dir,
                                               data_transforms_list=test_transforms_list,
                                               images_label_csv_path=self.opt.test_images_labels_path,
                                               normalize_mean=self.opt.normalize_mean,
                                               normalize_std=self.opt.normalize_std,
                                               batch_size=self.opt.batch_size,
                                               is_shuffle=False,
                                               num_works=self.opt.num_workers)
        return test_loader

    def generate_dataloader(self,
                            root_path,
                            data_transforms_list=None,
                            normalize_mean=None,
                            normalize_std=None,
                            images_label_csv_path=None,
                            batch_size=64,
                            is_shuffle=False,
                            num_works=4):
        # Train Data Set
        if normalize_mean is not None and normalize_std is not None:
            normalize = torchvision.transforms.Normalize(mean=normalize_mean, std=normalize_std)
            all_transforms = torchvision.transforms.Compose(data_transforms_list + [normalize])
        else:
            all_transforms = torchvision.transforms.Compose(data_transforms_list)
            # print("*"*100, all_transforms)

        obtained_dataset = dataset.ApplicationForm(root_path=root_path,
                                                   transform=all_transforms,
                                                   is_test=self.opt.is_test,
                                                   classes_index=self.opt.classes_index,
                                                   )

        obtained_loader = torch.utils.data.DataLoader(obtained_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=is_shuffle,
                                                      num_workers=num_works,
                                                      pin_memory=False)
        return obtained_loader

    @staticmethod
    def generate_dataloader_imagefolder(root_path,
                                        data_transforms_list=None,
                                        normalize_mean=None,
                                        normalize_std=None,
                                        batch_size=64,
                                        is_shuffle=False,
                                        num_works=4):
        # Train Data Set
        if normalize_mean is not None and normalize_std is not None:
            normalize = torchvision.transforms.Normalize(mean=normalize_mean, std=normalize_std)
            all_transforms = torchvision.transforms.Compose(data_transforms_list + [normalize])
        else:
            all_transforms = torchvision.transforms.Compose(data_transforms_list)

        obtained_dataset = dataset.ImageFolder()(root_path, transform=all_transforms)

        print(len(obtained_dataset))
        obtained_loader = torch.utils.data.DataLoader(obtained_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=is_shuffle,
                                                      num_workers=num_works,
                                                      pin_memory=True)
        return obtained_loader

