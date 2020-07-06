# -*- coding:utf-8 -*-

import os,sys
import warnings

import torch
import torch.utils.data

try:
    from config import opt
except:
    from .config import opt

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))


def pred_func(model, test_loader):
    images_name = []
    predict_labels = []
    pred_confidence_ls = []

    with torch.no_grad():
        return_pred_dict = {}
        for inputs, img_path in test_loader:
            # print(img_path)
            inputs = inputs.to(opt.device)

            _t = outputs = model(inputs)

            _outputs_softmax = torch.nn.functional.softmax(_t, dim=1)
            _t_pred_conf = float(_outputs_softmax[0][int(torch.max(_outputs_softmax, 1)[1].cpu().numpy())])  # 预测置信度
            pred_confidence_ls.append(_t_pred_conf)

            predict_results = torch.max(outputs, 1)[1].cpu().numpy()
            image_path = img_path[0].split('\\')[-3:]
            image_path = '\\'.join(image_path)
            images_name.append(image_path)

            predict_names = []


            for i in range(predict_results.shape[0]):
                one_predict = predict_results[i]
                to_classes = opt.index_classes[one_predict]

                to_str = ''.join(to_classes) if _t_pred_conf >= opt.threshold else "Others"
                predict_names.append(to_str)
            predict_labels.extend(predict_names)

            return_pred_dict[os.path.basename(img_path[0])] = predict_names[0]

    return return_pred_dict   # 返回一个字典{img_name: pred}