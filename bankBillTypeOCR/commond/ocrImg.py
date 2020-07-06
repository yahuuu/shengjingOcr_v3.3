#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
Created on 2018年8月7日

@author: wangs
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from commond import utils
from commond import InpaintLine
import cv2
import numpy as np

class ocrImg:
    def __init__(self, img, tessAPI):
        self.img = img
        self.getConfig()
        self.tessAPI = tessAPI
        
    def getConfig(self):
        #获取配置
        #红色印戳的颜色上线，0~10
        self.redMax = 2
        #是否去除红色印, 1 为去除， 0 为不去除
        self.isRemoveRed = 0
    
    def do_billType(self, isOSTU = False, isDilate = False):
        '''
        @attention: 去除红色的印戳, 在 OpenCV 的 HSV 格式中，H（色彩/色度）的取值范围是 [0，179]，S（饱和度）的取值范围 [0，255]，V（亮度）的取值范围 [0，255]
        红色区域为( (H >= 0  && H <= 10) || (H >= 125  && H <= 180)) && S >= 43 && V >= 46
        '''
        #转化为灰度图
        if len(self.img.shape) == 3:
            grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            grayImg = self.img.copy()
    
        H, W = self.img.shape[:2]
        #二值化图片
        if isOSTU:
            retval, gray = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)#ostu
        else:
            retval, gray = cv2.threshold(grayImg, 145, 255, cv2.THRESH_BINARY)#阈值
    
        #InpaintLine方法去除
        inpainted = InpaintLine.InpaintLine(gray, inpaintRadius=3, hor_ratio = 3)  
        tempImg = inpainted.copy()
    
        #去除噪点
        if isDilate:    
            #进行开运算
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            tempImg = cv2.morphologyEx(tempImg, cv2.MORPH_CLOSE,  element)
        #识别文字
        ocrImg = tempImg
        text = self.tessAPI.Tess_API_OCR_Image(ocrImg)
        text = str(text,encoding='utf-8')
        text = utils.handleText(text)  
        #text = unicode(text,"utf=8")
        return ocrImg, text    
    
    
    
       
    def do(self, isOSTU = False, isDilate = False):
        '''
        @attention: 去除红色的印戳, 在 OpenCV 的 HSV 格式中，H（色彩/色度）的取值范围是 [0，179]，S（饱和度）的取值范围 [0，255]，V（亮度）的取值范围 [0，255]
        红色区域为( (H >= 0  && H <= 10) || (H >= 125  && H <= 180)) && S >= 43 && V >= 46
        '''
        #转化为灰度图
        if len(self.img.shape) == 3:
            grayImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        else:
            grayImg = self.img.copy()
        #retval, grayImgOSTU = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)#ostu
        #grayImg = InpaintLine.InpaintLine(grayImg, inpaintRadius=3, hor_ratio = 10.1)
        H, W = self.img.shape[:2]
        #二值化图片
        if isOSTU:
            retval, gray = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)#ostu
        else:
            retval, gray = cv2.threshold(grayImg, 145, 255, cv2.THRESH_BINARY)#阈值
        #th2 = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,31,20)#自适应阈值
        #retval, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)#ostu
        
        #去掉下划线
#         himg = utils.horizonImg(gray, minLength = 200) #水平校正
#         iimg = utils.inPaintLine(himg, minLength = 300) #去除直线
#         iimg = utils.inPaintLine(iimg, minLength = 300) #去除直线
        #再次用InpaintLine方法去除
        inpainted = InpaintLine.InpaintLine(gray, inpaintRadius=3, hor_ratio = 3)  
        retval, gray = cv2.threshold(inpainted, 150, 255, cv2.THRESH_BINARY)#阈值
#         utils.pltShow([grayImg, iimg, himg, gray],['grayImg', 'iimg', 'himg', 'gray'])
        tempImg = gray.copy()
        
        
        #判断是否为空
        if utils.justBlankOrBlack(cv2.medianBlur(tempImg, 5), minR = 0.04):
            return tempImg, ''
        #去除噪点
        if isDilate:    
            #进行开运算
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            tempImg = cv2.morphologyEx(tempImg, cv2.MORPH_CLOSE,  element)
        #识别文字
        ocrImg = tempImg
        text = self.tessAPI.Tess_API_OCR_Image(ocrImg)
        text = utils.handleText(text)
        text = utils.handleZYtitle(text)
        if len(text) < 2:
            #print '-'*20,text
            ocrImg = inpainted
            text = self.tessAPI.Tess_API_OCR_Image(ocrImg)
        text = utils.handleText(text)
#         print 'text',text
#         utils.pltShow([gray, iimg, himg, ocrImg],['gray', 'iimg', 'himg', 'ocrImg'])        
        
        return ocrImg, text
    
    

    
                
if __name__ == '__main__':
    path  = r'D:\Users\jTessBoxEditorFX-2.0\title\zy_type_training\train\none\135263104-135263104-00000102455424973-5000.jpg'
    
#     img = cv2.imread(path)
#     ocri = ocrImg(img)
#     text = ocri.do()[-1]
#     print(text)
