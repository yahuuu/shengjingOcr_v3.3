# -*- coding:utf-8 -*-

import os,sys
import warnings

try:
    from networks.Network import Network
    from config import opt
    from data.GenerateDateLoader import GenerateDataLoader
    from predict_function import pred_func
except:
    from .networks.Network import Network
    from .config import opt
    from .data.GenerateDateLoader import GenerateDataLoader
    from .predict_function import pred_func

# from make_default_parameters import make_default_params


warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def main():
    # Network
    network = Network(opt)
    model = network.model
    # Dataset
    generate = GenerateDataLoader(opt)
    # print('Start to predict the data.')
    model, return_result_dict = network.load_model(model_path=opt.best_model_path,
                                                       return_list=['epoch', 'val_acc1', 'lr'])

    return model, return_result_dict, generate


