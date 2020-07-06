# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 13:49:27 2018

@author: houjinli
"""

import cv2
import numpy as np
import os
import glob


class TemplateMatch:
    def __init__(self,threshold=0.5):
        self.template_data = []
        self.modelType = []
        self.threshold = threshold
        
    def initModelByPath(self,modelPath,grayWay=0):
        data = {"zhczhqdqchxcd_tmpl.png":"zhczhqdqchxcd",
                "zhczhqdqchxcd_tmpl1.png":"zhczhqdqchxcd",
                "zhczhqtzhchxcd_tmpl1.png":"zhczhqtzhchxcd",
                "zhczhqtzhchxcd_tmpl.png":"zhczhqtzhchxcd"
                }
        # data = {"zhczhqdqchxcd_tmpl.png":"整存整取定期储蓄存单",
        #         "zhczhqtzhchxcd_tmpl.png":"整存整取特种储蓄存单"
        #         }
        # data1 = {"zhczhqdqchxcd_tmpl.png":"zhczhqdqchxcd",
        #         "zhczhqtzhchxcd_tmpl.png":"zhczhqtzhchxcd"
        #         }
        for modelFile in glob.glob(modelPath+"/*.png"):
            image = cv2.imread(modelFile)
            self.modelType.append(data[os.path.basename(modelFile)])
            if(grayWay==0):
                #灰度图
                image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                self.template_data.append(image) 
            else:
                #红色通道图
                image=image[:,:,2]
                self.template_data.append(image)                 
            
    def initModelByFileList(self,fileList):
        self.template_data = fileList
        
    def chageThreshold(self,threshold):
        self.threshold = threshold
        
    def match_process_one(self,gray,constant=30):
        result = []
        is_continue = True
        th_img=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, constant)
        row,col = gray.shape
        while(is_continue and constant > 5):
            for model in self.template_data:
                w, h = model.shape[::-1]
                if row < h or col < w:
                    continue
                model=cv2.adaptiveThreshold(model, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, constant) 
                res = cv2.matchTemplate(th_img,model,cv2.TM_CCOEFF_NORMED)
                if np.max(res) < self.threshold :
                    continue
                loc = np.where( res == np.max(res))
                for pt in zip(*loc[::-1]): 
                    result.append([pt[0],pt[1],w, h])
                is_continue = False
                break
            constant = constant-5
            th_img=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, constant)
        return  result 
    def findfitRegion(self, img):
        '''
        @attention: 从图片中获取含有字的部分
        @param img: 整张票据图片,3通道图片
        '''
        #读取图片并获取基本信息
        H, W = img.shape[:2]
        # 二值化 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        #去除噪点
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        dilation = cv2.dilate(binary, element, iterations = 1)
        #获取轮廓
        fcs = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(fcs) == 2:   contours = fcs[0]
        else:   contours = fcs[1]
        #utils.drawCountourAndShow(gray.copy(), contours, windowName = 'gray' )

        ys = []
        cns = []
        for posc, cnt in enumerate(contours):
            #计算该轮廓的面积
            area = cv2.contourArea(cnt) 
            if area / float(H * W) < 0.007:     continue #去除面积太小的
            # 轮廓近似，作用很小,将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定，使用的Douglas-Peucker算法，可以自己Google。
            #epsilon = 0.1 * cv2.arcLength(cnt, True)#轮廓周长
            #approx = cv2.approxPolyDP(cnt, epsilon, True)    
            # 找到最小的矩形，该矩形可能有方向,函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
            x, y, w, h = cv2.boundingRect(cnt)
            #if w / float(W) > 0.9 and y < int(W * 0.05):   continue #在顶部而且去除占用了整行的
            if w / float(W) < 0.15 :    continue #去除太短的
            ys.append(y)
            cns.append(cnt)

        try:
            ymin = min(ys)
            ori = img[max(ymin - 10, 0): ymin + int(0.30 * W), 0:W]
            #ori = img[max(ymin - 10, 0): ymin + int(0.35 * W), 0:W]
            #ori = img[max(ymin - 10, 0): ymin + int(0.55 * W), 0:W]
            return ori
        except ValueError:
            #print('min() arg is an empty sequence')
            return img    
    
    def getTopImg(self,img_total):
        H,W = img_total.shape[:2]
        #进行表头识别    
        #img = img_total[int(0.05 * H): int(0.35 * H), int(0.3 * W): int(0.85 * W)]
        img = img_total[int(0.05 * H): int(0.35 * H), int(0.2 * W): int(0.85 * W)]
        #img = img_total[int(0.05 * H): int(0.35 * H), 0: W]
        #img = img_total[int(0.01 * W): int(0.35 * H), int(0.01 * W): int(0.98 * W)]
    
        #找到合适的区域并保存
        img = self.findfitRegion(img)
        return img
        #cv2.imshow("img",img);cv2.waitKey(0)
           
    def match_opt_one(self,gray,constant=30): #return []  or [[x1,y1,w,h]]
        H,W = gray.shape
        gray = gray[0:int(H*0.25),0:int(W*1.0)]
        isContinue = True
        results = []
        image_bin=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, constant)
        row,col = gray.shape[::-1]
        while constant > 5 and isContinue:
            for i,tmpl_gray in enumerate(self.template_data):
                ###
                tmpl_bin=cv2.adaptiveThreshold(tmpl_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, constant)
                #cv2.imshow("image_bin",image_bin);cv2.waitKey(0)
                ###
                w, h = tmpl_bin.shape[::-1]
                if row<w or col<h:
                    continue
                res = cv2.matchTemplate(image_bin,tmpl_bin,cv2.TM_CCOEFF_NORMED) 
#                res = cv2.matchTemplate(image_bin,tmpl_bin,cv2.TM_CCOEFF) ##可解决 模板种类 匮乏 下的泛化能力
                
                min_res,max_res,_,(loc_col,loc_row) = cv2.minMaxLoc(res)
                if  max_res < self.threshold:
                    continue
                
                results.append([loc_col,loc_row,w,h,self.modelType[i],max_res])  ##x1,y1,w1,h1
                isContinue = False
                break
            constant = constant - 5
            image_bin=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, constant)
        
        if not len(results):
            return []
        results.sort(key=lambda x:x[-1],reverse=True)
        
        return results[0][:5]


class ModelMatchInter(TemplateMatch):
    def __init__(self, *args, **kwargs):
        super().__init__()
        modelPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
        self._tmple = TemplateMatch(threshold=0.6)
        self._tmple.initModelByPath(modelPath,grayWay=0)
        if self._tmple is None : print("none error")
    def get_class(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _location =  self._tmple.match_opt_one(gray)
        if len(_location)>0:
            x1, y1, w, h, type = _location
            print("__>", type)
            return type
        return "None"


if __name__ == '__main__':
    import shutil
    import time
    testPath = r"../../../test_img/065"

    def test_org():
        modelPath = r"./model"
        tmpl = TemplateMatch(threshold=0.6)
        tmpl.initModelByPath(modelPath,grayWay=0)
        for testFile in glob.glob(testPath+"/*[JPG,jpeg,jpg,png,JPEG,PNG]"):

            errorPath = os.path.join(testPath,"no")
            if not os.path.exists(errorPath):
                os.makedirs(errorPath)

            image = cv2.imread(testFile)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            start_time = time.time()
            location = tmpl.match_opt_one(gray)
            print(time.time()-start_time)
            if len(location)>0:
                x1,y1,w,h,type = location
                cutImg = gray[y1:y1+h,x1:x1+w]
                #cv2.imshow("cutImg",cutImg);cv2.waitKey(0)
                cv2.imwrite(testFile[:-4]+".png",cutImg)
                print(type)
            else:
                #return "None"
                print("None")
                #shutil.copyfile(testFile,os.path.join(errorPath,os.path.basename(testFile)))
    def test_new():
        import chardet
        model_match_inter = ModelMatchInter()
        for testFile in glob.glob(testPath+"/*[JPG,jpeg,jpg,png,JPEG,PNG]"):
            img = cv2.imread(testFile)
            result = model_match_inter.get_class(img)
            # print(chardet.detect(result))
            print(result)


    test_new()




































