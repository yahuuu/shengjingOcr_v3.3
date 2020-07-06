#coding:utf-8

# python3.6.2

import sys, os
import cv2
import torch
import cv2 as cv
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.my_loader import data_loadermy, data_loadermy1


try:
    from main_pred import *
except:
    from .main_pred import *

warnings.filterwarnings('ignore')

####置为全局变######## just 4 interface
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else "-1"
model, return_result_dict, generate = main()
####################

def pred(img_cv_bgr, img_abspth=" "):
    """

    :param img_cv_bgr:  单图np
    :param img_abspth:  可不传递路径
    :return:  IDCardBack:unidode 或者 IDCardFront:str 或 Others:str 或 None:str
    """
    try:
        # test_loader = generate.test_generation(img_abspth)

        img_pil = Image.fromarray(cv2.cvtColor(img_cv_bgr, cv2.COLOR_BGR2RGB))
        test_loader = data_loadermy1(img_pil, img_abspth)

        pred_dict = pred_func(model, test_loader)
        pred_dict = list(pred_dict.values())[0]  # 4 py3
        return pred_dict
    except Exception as e:
        print(e)
        return "None"


if __name__ == "__main__":
    import time
    def func_time(func):
        def inner(*args, **kw):
            start_time = time.time()
            func(*args, **kw)
            end_time = time.time()
            print('Time', end_time - start_time, 's')
        return inner

    @func_time
    def single(img_abspth):
        img_cv = cv2.imread(img_abspth, cv2.IMREAD_COLOR)
        rlt = pred(img_cv_bgr=img_cv, img_abspth=" ")
        print(rlt, "->", "/".join(img_abspth.split(os.path.sep)[-3:]),"->", end="")

    def multiple():
        rpth = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_img")
        for root, dir, imls  in os.walk(rpth):
            if not imls: continue
            for im in imls:
                single(os.path.join(root, im))

    # single()
    multiple()
