# -*- coding:utf-8 -*-

import datetime
import os
import warnings
import torch


class DefaultConfig(object):
    # add new threshold
    threshold= 0.8  # a threshold 4 others labels

    # ======About path======
    # project_path = r"/media/ApplicationFormClassification"
    project_path = os.path.dirname(os.path.abspath(__file__))

    save_model_path = os.path.join(project_path, 'checkpoints')

    data_path = os.path.join(project_path, 'dataset', 'ApplicationForm')
    train_images_labels_path = None
    val_images_labels_path = None

    train_data_dir = os.path.join(data_path, 'train')
    val_data_dir = os.path.join(data_path, 'val')

    test_dir = os.path.join(data_path, 'test') # no used
    test_images_labels_path = None
    best_model_path = os.path.join(project_path, "checkpoints", "resnet18_27.t7")

    # ======About the dataset======
    dataset_name = 'ApplicationForm'
    normalize_mean = None
    normalize_std = None
    num_workers = 1


    # ----------
    # classes_index = {'201004': 7, '201002': 5, '201001': 4, '201003': 6, '101006': 3, '101002': 1, '101003': 2,
    #                  '101001': 0}
    # index_classes = {3: '101006', 2: '101003', 5: '201002', 7: '201004', 1: '101002', 6: '201003', 0: '101001',
    #                  4: '201001'}
    classes_index = {"IDCardFront":0, "IDCardBack":1, "Others":2}
    index_classes = {0:"IDCardFront", 1:"IDCardBack", 2:"Others"}
    # ----------

    num_classes = len(classes_index.keys()) 
    assert len(classes_index.keys()) == len(index_classes.keys()), "config error"

    # ======About the train phrase======
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpus = "cpu"
    batch_size = 1

    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-5
    is_resume = False
    is_test = True

    start_epoch = 0
    end_epoch = 200 

    best_acc = 0
    best_epoch = 0
    #
    # =====About the network parameters=====
    pretrained_model_path = None
    network_type = 'resnet18'
    image_size_width = 224
    image_size_height = 224
    suffix = None
    time_stamp = '{0:%Y-%m-%d-%H-%M}'.format(datetime.datetime.now())

    def parse(self, kwargs_dict):
        """
        update the parameters according the kwargs
        """
        for k, v in kwargs_dict.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute %s" % k)
            setattr(self, k, v)

    def print_configuration(self):
        print('user configuration:')
        configuration_dict= self.__class__.__dict__
        configuration_dict = dict(sorted(configuration_dict.items(), key=lambda x: x[0]))
        for k, v in configuration_dict.items():
            if (not k.startswith('__')) and k != 'parse' and k != 'print_configuration':
                print('{}:{}'.format(k, getattr(self, k)))


opt = DefaultConfig()

