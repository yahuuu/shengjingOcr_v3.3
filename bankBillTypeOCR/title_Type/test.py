# -*- coding: utf-8 -*-
'''
@attention: 
@author: wsd
'''
import os
import sys
import cv2
import glob
import numpy as np
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
#from commond import utils
#from commond import ocrImg
#import re
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../tesseract_reg_online'))
#from denoise_recog import vertical_reg

def drawCountourAndShow(img, contours, windowName = 'img', color = 255 ):
    '''
    @attention: 在图片上画出轮廓
    '''
    cv2.drawContours(img, contours, -1, (color), 2 )
    cv2.imshow(windowName,img)
    cv2.waitKey(0)

   
if __name__ == '__main__':
    testPath = "../../test"
    #imgList = [
        ##r"../../test\2.JPG",
        ##r"../../test\2a0c3d8c0b6c40de85596a8cf83edfef.JPG",
        ##r"../../test\3c87a45e4cc34b80902028f0d5b454ef.JPG",
        ##r"../../test\462b54e27f0949aba1f4f118288fe680.JPG",
        ##r"../../test\90b699790b6a4355bcd0308e7496017e.JPG",
        ##r"../../test\aa1e009630d54cfcad4fc1812cb8df34.JPG",
        ##r"../../test\f841539d35094ebdbc804957fb9ad473.JPG",
        ##r"../../test\fade8841da3f4ef882f55ea8ad29b836.JPG",
        
        #r"../../test\90b699790b6a4355bcd0308e7496017e.JPG",
    #]
    imgList = glob.glob(testPath+"/*.JPG")
    for imgPath in imgList:
        print(imgPath)
        ori_img = cv2.imread(imgPath)
        H, W = ori_img.shape[:2]
        #切分竖排字
        imges=[]
        sub_label = [u"第一联",u"第二联",u"第三联"]
        img_cut = ori_img[int(0.3* H): int(0.9 * H), int(0.9 * W): int(0.98 * W)]
        gray_img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
        H0,W0 = img_cut.shape[:2]
        ret, binary = cv2.threshold(gray_img_cut, 140, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,int(H0*0.06)))
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,int(0.25*H0)))
        binary_median = cv2.medianBlur(binary, 3)#中值滤波
        gray = cv2.dilate(binary_median, kernel, iterations = 1)
        fcs = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL cv2.RETR_TREE
        if len(fcs) == 2:   contours = fcs[0]
        else:   contours = fcs[1]
        #drawCountourAndShow(img_cut.copy(), contours, windowName = 'gray' )
        for posc, cnt in enumerate(contours):
            #计算该轮廓的面积
            area = cv2.contourArea(cnt)
            #x, y, w, h = cv2.boundingRect(cnt)
            #ocrORI = img_cut[y:y + h,x:x+w]
            #print(y)
            #cv2.imshow("ocrORI",ocrORI);cv2.waitKey(0)            
            if area / float(H0 * W0) < 0.03 or area / float(H0 * W0) > 0.1:  continue
            x, y, w, h = cv2.boundingRect(cnt)
            #ocrORI = ~binary[y:y + h, max(x - 4, 0):x+w+4]
            ocrORI = ~binary[y+4:y + h-4, max(x - 2, 0):x+w+2]
            #print(y)
            #cv2.imshow("ocrORI",ocrORI);cv2.waitKey(0)
            imges.append([y, ocrORI, x + y])
        imges.sort(key=lambda imges:imges[0], reverse=False)
        
        if len(imges)>0:
            listType= [-1]*4
            tmp_img = imges[0][1]
            for m_imagePath in glob.glob("../../tmpl_model/*.png"):
                type = os.path.basename(m_imagePath).replace(".png","")
                m_image = cv2.imread(m_imagePath,0)
                h, w = m_image.shape[:2]
                res_ = cv2.matchTemplate(tmp_img,m_image,cv2.TM_CCOEFF_NORMED)
                max_res = np.max(res_)
                listType[int(type)]=max_res
            print(np.argmax(listType))
            cv2.imshow("tmp_img",tmp_img);cv2.waitKey(0)
                