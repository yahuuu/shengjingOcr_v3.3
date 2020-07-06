# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 16:00:56 2018

@author: rcx
"""

import cv2
import numpy as np
import sys
import os


'''
主函数：InpaintLine
功 能： 去除水平线
输 入： 灰度图，np.uint8,
输 出:  inpaint 过后的灰度图
步 骤： 
    1. 用形态学 得到水平线的位置
    2. 膨胀 得到 mask
    3. inpaint 
    4. 针对inpaint 不能处理边界的bug，后处理
    
cv2.inpaint  个人使用总结、记录：
    1. mask 掩模 为 255的地方需要用半径为 3 (此方法内inpaintRadius = 3)的邻域像素 来推理得到
    2. 如果 mask  第0行 为255， cv2.inpaint 并不能处理，（即边界 存在bug）
'''

show_img_tmp = 0
filename = ''

def get_mask(im_bin,hor_ratio = 10.1):  ##将im_bin 的水平线 当做待修复的区域，生成掩模
    row,col = im_bin.shape    
       
    hor_size = int(round(col / hor_ratio))
    if hor_size % 2 ==0:
        hor_size += 1
        
    kernel0=cv2.getStructuringElement(cv2.MORPH_CROSS,(hor_size,1))
    morphologyImg0=cv2.morphologyEx(im_bin,cv2.MORPH_CLOSE,kernel0,iterations=1)
    
    return ~morphologyImg0

def InpaintLine(im_gray,inpaintRadius=3,hor_ratio = 10.1):
    import matplotlib.pyplot as plt
    row_gray,col_gray = im_gray.shape
    im_rl  =  im_gray.copy()     
    #im_rl = removeLight(im_gray)
    im_bin = Ot_BeresenThreshold_fast(im_rl)

    if show_img_tmp:
        plt.imshow(im_bin,'gray');plt.title('im_bin');plt.show()
    
    mask = get_mask(im_bin,hor_ratio = hor_ratio)
    
    if not np.count_nonzero(mask):
        return im_gray
    
    kernel0  = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    
    mask = cv2.dilate(mask,kernel0)
    
    if show_img_tmp:
        plt.imshow(mask,'gray');plt.title('mask');plt.show()
                     
    inpainted = cv2.inpaint(im_gray,mask,inpaintRadius,cv2.INPAINT_TELEA)  
#    inpainted = np.array(inpaint(im_gray,mask))   ##add_by _ ywj 
    
    #解决 cv2.inpaint 不能处理边界的bug

    cols = np.nonzero(mask[0])[0]  ##第一行 mask为255 的列坐标
    if len(cols):
        w_tmp = inpaintRadius
        im2 = im_gray[:1+w_tmp].copy()
        im_white = np.zeros((2*w_tmp+1,col_gray+2*w_tmp),np.uint8)
        im_white[w_tmp:,w_tmp:-w_tmp] = im2
        sum_move = (cv2.blur(np.float32(im_white),(1+2*w_tmp,1+2*w_tmp))*(1+2*w_tmp)**2)[w_tmp]
        CT = []
        for x_ in cols:
            ## 因为左右边界，计算宽*高 时 需要减去 的列数(宽)： max(w_tmp-x_,0,w_tmp - (col_gray -1- x_))
            value = sum_move[x_+w_tmp]/((1+w_tmp)*(2*w_tmp+1 - max(w_tmp-x_,0,w_tmp - (col_gray -1- x_))))
            CT.append(value)
        inpainted[0][cols] =  np.uint8(np.round(CT))

        

    if show_img_tmp:    ##为了展示 识别前 放进去的二值图 是怎样
#        cv2.imwrite(os.path.splitext(filename)[0]+'inpaint'+'.png',inpaint)
        inpaint_rl = removeLight(inpainted)
        im_inpaint_bin = Ot_BeresenThreshold_fast(inpaint_rl)
        
        row,col = im_inpaint_bin.shape
        white_width = max(int(round(min(row,col) * 0.2)),4)  #
        im_white = ~np.zeros((row+2*white_width,col+2*white_width),np.uint8)
        im_white[white_width:white_width+row,white_width:white_width+col] = im_inpaint_bin.copy()
        im_inpaint_bin = im_white.copy()
        
        plt.imshow(im_inpaint_bin,'gray') ;plt.title('im_inpaint_bin'); plt.show()
        
    return inpainted
    
'''
功能：快速二值化
输入参数：image----灰度图像路径
'''
def Ot_BeresenThreshold_fast(img,C1 = -0.02,C2=0.02):
    rows,cols=img.shape
    ret,_ = cv2.threshold(img,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
 
    beta_lowT = (1+C1)*ret
    beta_highT = (1+C2)*ret
    _,out1=cv2.threshold(img,beta_lowT,255,cv2.THRESH_BINARY)
    _,out2=cv2.threshold(img,beta_highT,255,cv2.THRESH_BINARY)
 
 
    CT=cv2.blur(np.float32(img),(5,5))  ##等价于5*5窗口
 
    out3=(img-CT)*((~out2//255)*(out1//255))
    out3[out3 >= 0]=1
    out3[out3<0]=0   
    out3=np.uint8(np.round(out3))
 
    return out3*out1    


if __name__ == '__main__':
    show_img_tmp = 1
#    filename_mask ='D:/ZYData/Debug/InpaintLine/_img_white.bmp'  ##生成 膨胀 和闭合的 mask
#    mask = cv2.imread(filename_mask,cv2.IMREAD_UNCHANGED)
##    cv2.imwrite(os.path.splitext(filename_mask)[0]+'_tmp.png',mask)
#    kernel0  = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))  
#    mask_dilate = ~cv2.dilate(~mask,kernel0)
#    cv2.imwrite(os.path.splitext(filename_mask)[0]+'_dilated.bmp',mask_dilate)
#    
#    mask_close=~cv2.morphologyEx(~mask,cv2.MORPH_CLOSE,kernel0,iterations=1)
#    cv2.imwrite(os.path.splitext(filename_mask)[0]+'_closed.bmp',mask_close)
    
    
    
    
    filename ='D:/ZYData/Debug/DSCC_EHW_Debug/Cut/20180706134220_70.png' 
    #filename ='D:/ZYData/Debug/DSCC_EHW_Debug/Cut/20180706134220_70_ELine.png' 
    
    #filename ='D:/ZYData/Debug/InpaintLine/image0000010A_20.png'
    filename_gray ='D:/ZYData/Debug/InpaintLine/image0000010A_20_gray.png'
    
#    filename ='D:/ZYData/Debug/InpaintLine/image0000010A_70.png'
#    filename_gray ='D:/ZYData/Debug/InpaintLine/image0000010A_70_gray.png'
    
#    filename ='D:/ZYData/Debug/InpaintLine/20180706134220_70.png'
#    filename_gray ='D:/ZYData/Debug/InpaintLine/20180706134220_70_gray.png'
    
    
#    im_gray = cv2.imread(filename_gray,cv2.IMREAD_UNCHANGED)
#    inpainted = InpaintLine(im_gray)
    
    filename_gray = r'D:/ZYData/ZYOCR/demos/receipt/test_data/skew/image.png'
        
    filename_gray = './00000345.png'
    filename_gray = './ff80808164b22a2601650dcbafc2794b_00000079.png'
    filename_gray = './ff80808164b22a2601650de70b070237_00000239.png'
    filename_gray = './ff80808164b22a26016512d3eb7e7e0a_00000673.png'
    
#    im_gray = cv2.imread(filename_gray,cv2.IMREAD_UNCHANGED)
    im_gray = cv2.imread(filename_gray,0)
#    mask = cv2.imread(r'D:/ZYData/ZYOCR/demos/receipt/test_data/skew/mask2.png',cv2.IMREAD_UNCHANGED)
#    inpaintRadius = 3
#    inpainted = cv2.inpaint(im_gray,mask,inpaintRadius,cv2.INPAINT_TELEA)  
    inpainted = InpaintLine(im_gray,hor_ratio = 9.1)
    cv2.imwrite(os.path.splitext(filename_gray)[0]+'_'+'inpainted.png',inpainted) 
    
    
    





