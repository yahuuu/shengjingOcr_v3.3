# -*- coding:utf-8 -*-

u'''
票据的类型:
如果返回'None'表明没有判断出来类型
'''
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from billTitleOCR import ocrBillTitle
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from commond import utils


def billType(imgPath, tessApi,tess_api_vert,modelImgList):
    '''
    @attention: 票据类型的识别接口
    @param imgPath:图片路径， 或者图片
    @param tessApi:tesseract 的票据分类语言包api对象
    @return: 返回字符串unicode的票据类型
    '''
    ob = ocrBillTitle(imgPath, tessApi,tess_api_vert,modelImgList)
    res = ob.do()
    
    return res[2]
    
if __name__ == '__main__':
    import time
    start = time.time()
    
    #导入tesseract的api
    sys.path.append(u'../../ocr_models/Tesseract_API')
    from TesseractAPI_SingleHandle_Class import TessAPI
    import cv2
    import numpy as np
    
    tess_api = TessAPI()
    tess_api.Tess_API_Init(lang = 'chi_new_stsong_jx',flag_digit = 0,psm = 6)
    tess_api_vert = TessAPI()
    tess_api_vert.Tess_API_Init(lang='chi_new_stsong_jx', flag_digit=0, psm=5)    
    
    #imgPath = unicode(r"D:\01work\python\test\GZYH_OCR\testImg\test01.jpg".decode('utf-8'))
    imgPath = r"D:\01work\python\test\GZYH_OCR\test\test_shenjing\test01\0befa76be3364c63818abe30f589d07a.JPG"
    #img = cv2.imdecode(np.fromfile(imgPath, dtype = np.uint8), -1)
    #result = tess_api.Tess_API_OCR_Image(img)
    start = time.time()
    res = billType(imgPath, tess_api,tess_api_vert)
    print(res)
    #print 'time:', time.time() - start
