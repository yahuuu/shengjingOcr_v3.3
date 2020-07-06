# -*- coding:utf-8 -*-
import os
import sys
import torch
from torchsummary import summary

import models


class Network(object):
    def __init__(self, opt):
        super(Network, self).__init__()
        # print("opt.num_classes", opt.num_classes)
        self.model = getattr(models, opt.network_type)(num_classes=opt.num_classes)
        self.opt = opt
        self.device = opt.device
        self.model_save_list = ['model', 'val_acc1', 'val_acc5', 'lr', 'epoch', 'optimizer', 'batch_size']

        self.move_model()

    def summary_model(self):
        """
        Print the whole network.

        :return
        """
        summary(self.model,
                input_size=(3, self.opt.image_size_width, self.opt.image_size_height),
                device='cpu')

    def move_model(self):
        if self.device.type == 'cuda':
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(device=self.device)
        return self.model

    def load_model(self, model_path, return_list=None):
        """
        Load the pre-trained model weight

        :return:
        """
        checkpoint = torch.load(model_path, map_location='cpu') # only 4 cpu machine

        # if os.system("nvidia-smi")!=0 or self.opt.gpus.lower() =='cpu': # 4 cpu machine or only use cpu env
        try:
            # if self.opt.gpus.lower() =='cpu': # 4 cpu machine or only use cpu env
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model"].items():
                name = k[7:] # remove `module.`
                # name = "module."+ k # add `module.`
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=True)
        except:
            self.model.load_state_dict(checkpoint["model"], strict=True)

        # elif os.system("nvidia-smi") == 0: # 4 gpu machine
        #     self.model.load_state_dict(checkpoint["model"], strict=True)

        return_result_dict = {}

        for key in return_list:
            if key not in self.model_save_list:
                raise KeyError('The pre-trained model has not the {}'.format(key))
            else:
                return_result_dict[key] = checkpoint[key]

        return self.model, return_result_dict

    def save_model(self, model, optimizer,
                   current_epoch, train_loss,
                   val_acc1, val_acc5=None,
                   is_save_each_epoch=False):

        current_lr = optimizer.param_groups[0]['lr']
        state = {'model': model.state_dict(),
                 'val_acc1': val_acc1,
                 'val_acc5': val_acc5,
                 'lr': current_lr,
                 'epoch': current_epoch,
                 'optimizer': optimizer,
                 'batch_size': self.opt.batch_size}

        detail_path = os.path.join(self.opt.save_model_path,
                                   '{}_{}'.format(self.opt.dataset_name, self.opt.network_type),
                                   '{}_{}'.format(self.opt.time_stamp, self.opt.suffix))
        if not os.path.exists(detail_path):
            os.mkdir(detail_path)

        model_name = 'epoch_{}_train_loss_{:.5f}_acc_{:.4f}_{}_{}.t7'.format(current_epoch, train_loss, val_acc1,
                                                                             self.opt.network_type, self.opt.suffix)

        if val_acc1 >= self.opt.best_acc: # and val_acc1 > 80:
            print('saving the better model ... ...')

            # Update the values of the opt object
            self.opt.parse({'best_epoch': current_epoch,
                            'best_acc': val_acc1})

            model_directory = os.path.join(detail_path, 'best_model')
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

            full_model_path = os.path.join(model_directory, model_name)
            torch.save(state, full_model_path)

        if is_save_each_epoch:
            each_full_model_path = os.path.join(detail_path, model_name)
            torch.save(state, each_full_model_path)
