# -*- coding: utf-8 -*-
'''
@attention: 
@author: wsd
'''
import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from commond import utils
from commond import ocrImg
import re
#sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../tesseract_reg_online'))
#from denoise_recog import vertical_reg
    

class ocrBillTitle():
    '''
    @attention: 找出票据上部的抬头，用于票据的分类。
    @param imgPath: 输入的图片
    @param tessZYAPI: tesseract API for ZY bill
    @param tessTYAPI: tesseract API for TY bill
    '''
    def __init__(self, imgPath,tessAPI = '',tess_api_vert='',modelImgList=[]):
        #读取配置信息    @param types: 发票类型列表 @param typeRatios: 发票类型的高宽比率
        fileDir = os.path.join(os.path.dirname(os.path.abspath(__file__)) , './')
        self.types = utils.getTypes(fileDir + 'billTypeInfo.cfg')
        self.imgPath = imgPath
        #self.modelImgList = [os.path.join(fileDir+'tmpl_model',itemPath) for itemPath in os.listdir(fileDir+'tmpl_model') if itemPath.endswith(".png")]
        self.modelImgList = modelImgList
        self.tessAPI = tessAPI
        self.tess_api_vert = tess_api_vert
        
    def fuzzyMatch(self,ocrtext):
        #计算概率，选取最大的概率值
        maxRatio, proKey, proValue = 0.0, u'None', u'None'
        for item in self.types:
            lenKey, key, value = item[0], item[1], item[2]
            if len(key)>4:
                num = 0.0
                tstr = ocrtext
                akey = u''
                for pk, ki in enumerate(key):
                    if ki in tstr:   
                        num += 1
                        ind = tstr.index(ki)
                        tstr = tstr[ind:]
                        akey+=ki
                    akey += u'.{0,3}'
                ratio = num / len(key)
                if ratio > 0.7 and ratio > maxRatio and len(re.findall(akey, ocrtext)) > 0:    
                    maxRatio, proKey, proValue = ratio, key, value
        if maxRatio > 0.7: return proValue
        else:
            return None
    
    def findContourAndgetText(self, img_total):
        '''
        @attention: 根据轮廓截取相应的区域，并识别这些区域
        @param img: 输入的图片，3通道图片
        '''
        imges = self.getTopImg(img_total)  
        #由上至下排序
        imges.sort(key=lambda imges:imges[0], reverse=False)    
        ocr_img,ocrtext, res = u"None",u"None",u"None"
        
        ocrORI = None
        tmp_ocrtext = None 
        
        count = 0
        
        #不去除噪点
        for pos, ig in enumerate(imges):
            H, W = ig[1].shape[:2]
            if pos >= 4:    continue

            ocrImgs = utils.getDiffColorImg(ig[1])
            
            
            
            #识别并处理文字
            for oi in ocrImgs:
                ocr = ocrImg.ocrImg(oi, self.tessAPI)
                ocrORI, tmp_ocrtext = ocr.do_billType(isOSTU = True, isDilate = False)
                tmp_ocrtext = re.sub(u'托收.{0,1}证', u'托收凭证', tmp_ocrtext)
                #print(tmp_ocrtext)
                #cv2.imshow('oi',ocrORI);cv2.waitKey(0)
                #判断字符串res属于那种标题
                for item in self.types:
                    key, value = item[1], item[2]
                    if key in tmp_ocrtext:      
                        res = value
                        ocr_img = ocrORI
                        ocrtext = tmp_ocrtext
                        break
                tmp = self.fuzzyMatch(tmp_ocrtext)
                if tmp != None:
                    res = tmp
                    ocr_img = ocrORI
                    ocrtext = tmp_ocrtext                    
                if isinstance(ocr_img, np.ndarray):
                    break
            if isinstance(ocr_img, np.ndarray):
                break            
        if not isinstance(ocr_img, np.ndarray):
            ocr_img = ocrORI
            ocrtext = tmp_ocrtext
            res = u"None"
        return ocr_img,ocrtext,res
            
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
        #element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        #erosion = cv2.erode(binary, element, iterations = 1)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (70, 1))
        dilation = cv2.dilate(binary, element, iterations = 1)
        #获取轮廓
        fcs = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(fcs) == 2:   contours = fcs[0]
        else:   contours = fcs[1]
        #utils.drawCountourAndShow(gray.copy(), contours, windowName = 'gray' )
        
        ys = []
        # cns = []
        for posc, cnt in enumerate(contours):
            #计算该轮廓的面积
            # area = cv2.contourArea(cnt)
            #if area / float(H * W) < 0.007:     continue #去除面积太小的
            # 轮廓近似，作用很小,将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定，使用的Douglas-Peucker算法，可以自己Google。
            # epsilon = 0.1 * cv2.arcLength(cnt, True)#轮廓周长
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # 找到最小的矩形，该矩形可能有方向,函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
            x, y, w, h = cv2.boundingRect(cnt)
            #if w / float(W) > 0.9 and y < int(W * 0.05):   continue #在顶部而且去除占用了整行的
            if w / float(W) < 0.15 :    continue #去除太短的
            ys.append(y)
            # cns.append(cnt)
            
        try:
            ymin = min(ys)
            ori = img[max(ymin - 10, 0): ymin + int(0.30 * W), 0:W]
            return ori
        except ValueError:
            #print('min() arg is an empty sequence')
            return img

    
    def getTopImg(self,img_total):
        H,W = img_total.shape[:2]
        #进行表头识别    
        img = img_total[int(0.01 * W): int(0.35 * H), int(0.01 * W): int(0.98 * W)]
            
        #找到合适的区域并保存
        img = self.findfitRegion(img)
        #找到直线（可能是下划线或者是表格线)，然后进行旋转校正和截取合格的区域
        img = utils.oriByTopLines(img,minLength = int(W * 0.45),isCut=False)
        img = utils.oriByTopLines(img,minLength = int(W * 0.30),isCut=False)
        # H,W = img.shape[:2]
                #cv2.imshow('11', img)
        #cv2.waitKey(0)
        #读取图片并获取基本信息
        H, W = img.shape[:2]
        # 二值化 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)#二值化
        #  binary = cv2.adaptiveThreshold(gray, 255,
        #                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 30)#二值化
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (int(W * 0.05), 1))#cv2.MORPH_RECT  W * 0.03 由于银行本票字间距过大调整到0.5
        
        #binary_median = cv2.medianBlur(binary, 3)#中值滤波
        binary_median = cv2.medianBlur(binary, 3)#中值滤波
        gray = cv2.dilate(binary_median, kernel, iterations = 1)
        #cv2.imshow("gray", gray);cv2.waitKey(0)
        #获取轮廓
        fcs = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL cv2.RETR_TREE
        if len(fcs) == 2:   contours = fcs[0]
        else:   contours = fcs[1]
        #utils.drawCountourAndShow(img.copy(), contours, windowName = 'gray' ) 
        #获取相应的ori并由上至下排序
        imges = []
        for posc, cnt in enumerate(contours):
            #计算该轮廓的面积
            area = cv2.contourArea(cnt)
            if area / float(H * W) < 0.015:  continue
            # 轮廓近似，作用很小,将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定，使用的Douglas-Peucker算法，可以自己Google。
            # epsilon = 0.1 * cv2.arcLength(cnt, True)#轮廓周长
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # 找到最小的矩形，该矩形可能有方向,函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.imshow("img_total",img[y:y+h,x:x+w]);cv2.waitKey(0)
          
            if w / float(W) < 0.12 or w / float(W) > 0.9: continue #行宽不够的说明不是标题

            if h<16: continue   #由0.6 换成 0.7 通用凭证-00000019.JPG  现金支票 删除h > H*0.7 or
            #太细的则去除
            if float(w) / h > 35 or float(h) / w > 35:  continue
            #获取box区域并且识别处        理文字
            ocrORI = img[max(y - 2, 0):y + h + 2, x:x+w]
            
            ho,wo = ocrORI.shape[:2]
           
            if np.sum(binary[y:y+h, x:x+w]/255)*1.0/(ho*wo)>0.27 and w*1.0/h<4:continue
            if np.sum(binary[y:y+h, x:x+w]/255)*1.0/(ho*wo)<0.06 : continue #由0.1 改为0.06  资信证明书（正本） 类型与旁边的红色戳连接在一起 使面积变大黑色比例变小

            if ho < 43: ocrORI = img[max(y - 2, 0):y + int(ho * 1.2), x:x+w]
            
            imges.append([y, ocrORI, x + y])  
        return imges
    
    def getmiddileImg(self,img):
        H,W = img.shape[:2]
        #img = img[int(0.37*H):int(0.45*H),int(0.2*W):int(0.8*W)]
        img = img[int(0.34*H):int(0.45*H),int(0.2*W):int(0.8*W)]
        #读取图片并获取基本信息
        H, W = img.shape[:2]
        # 二值化 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)#二值化
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (int(W * 0.08), 1))#cv2.MORPH_RECT  W * 0.03
    
        binary_median = cv2.medianBlur(binary, 3)#中值滤波
        gray = cv2.dilate(binary_median, kernel, iterations = 1)     
        #cv2.imshow("gray", gray);cv2.waitKey(0)
        #获取轮廓
        fcs = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL cv2.RETR_TREE
        if len(fcs) == 2:   contours = fcs[0]
        else:   contours = fcs[1]
        #utils.drawCountourAndShow(img.copy(), contours, windowName = 'gray' ) 
        #获取相应的ori并由上至下排序
        imges = []
        for posc, cnt in enumerate(contours):
            #计算该轮廓的面积
            area = cv2.contourArea(cnt)
            if area / float(H * W) < 0.08:  continue
            # 轮廓近似，作用很小,将轮廓形状近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来决定，使用的Douglas-Peucker算法，可以自己Google。
            # epsilon = 0.1 * cv2.arcLength(cnt, True)#轮廓周长
            # approx = cv2.approxPolyDP(cnt, epsilon, True)
            # 找到最小的矩形，该矩形可能有方向,函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
            x, y, w, h = cv2.boundingRect(cnt)
        
            if w / float(W) < 0.15 or w / float(W) > 0.9: continue #行宽不够的说明不是标题
        
            #太细的则去除
            if float(w) / h > 35 or float(h) / w > 35:  continue
            #获取box区域并且识别处        理文字
            ocrORI = img[max(y - 2, 0):y + h + 2, x:x+w]
        
            imges.append([y, ocrORI, x + y]) 
        return imges

    def do(self):
        '''
        @attention: 从图片中获取单个字的坐标
        @param img: 整张票据图片
        '''   
        ocrtext, restext =  u'None',u'None'
        ocrtext_ty, restext_ty = u'None',u'None'
        
        if not isinstance(self.imgPath, np.ndarray):#如果不是图片则是路径
            #读取图片路径
            ori_img = cv2.imdecode(np.fromfile(self.imgPath, dtype = np.uint8), -1)
        else:
            ori_img = self.imgPath 
        H, W = ori_img.shape[:2]
        #print(H,W,float(W)/H)    
        #根据轮廓识别文字
        img,ocrtext, restext = self.findContourAndgetText(ori_img)
        if restext == u"结算业务申请书":
            img_cut = ori_img[int(0.1* H): int(0.18 * H), int(0.68 * W): int(0.9 * W)]
            img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
            #retval, binary = cv2.threshold(img_cut, 150, 255, cv2.THRESH_BINARY)
            binary = cv2.adaptiveThreshold(img_cut, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 20)
            if utils.justBlankOrBlack(binary, minR = 0.02):
                restext = u"结算业务申请书（无号码）"
            
            ##切分竖排字
            #imges=[]
            #img_cut = ori_img[int(0.3* H): int(0.9 * H), int(0.9 * W): int(0.98 * W)]
            #gray_img_cut = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
            #H0,W0 = img_cut.shape[:2]
            #ret, binary = cv2.threshold(gray_img_cut, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
            #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4,int(H0*0.06)))
            ##kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1,int(0.25*H0)))
            #binary_median = cv2.medianBlur(binary, 3)#中值滤波
            #gray = cv2.dilate(binary_median, kernel, iterations = 1)
            #fcs = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#cv2.RETR_EXTERNAL cv2.RETR_TREE
            #if len(fcs) == 2:   contours = fcs[0]
            #else:   contours = fcs[1]
            ##utils.drawCountourAndShow(img_cut.copy(), contours, windowName = 'gray' )
            #for posc, cnt in enumerate(contours):
                ##计算该轮廓的面积
                #area = cv2.contourArea(cnt)
                #if area / float(H0 * W0) < 0.03 or area / float(H0 * W0) > 0.1:  continue
                ##if area / float(H0 * W0) < 0.14 or area/ float(H0 * W0) > 0.25:  continue
                #x, y, w, h = cv2.boundingRect(cnt)
                ##ocrORI = img_cut[y:y + h, max(x - 4, 0):x+w+4]
                #ocrORI = ~binary[y+4:y + h-4, max(x - 2, 0):x+w+2]
                ##cv2.imshow("ocrORI",ocrORI);cv2.waitKey(0)
                #imges.append([y, ocrORI, x + y])
            #imges.sort(key=lambda imges:imges[0], reverse=False)
            
            ##imgList = [res[1] for res in imges]
            ##sub_label = vertical_reg(imgList,self.tess_api_vert)
            
            #sub_label = None
            #sub_labelList = [None,u"第一联",u"第二联",u"第三联"]
            #if len(imges)>0:
                #listType= [-1]*4
                #tmp_img = imges[0][1]
                #col,row = tmp_img.shape[:2]
                #for type,m_image in self.modelImgList:
                    #h, w = m_image.shape[:2]
                    #if h>col or w>row:
                        #continue
                    #res_ = cv2.matchTemplate(tmp_img,m_image,cv2.TM_CCOEFF_NORMED)
                    #max_res = np.max(res_)
                    #listType[int(type)]=max_res
                #if max(listType) != -1:
                    #sub_label = sub_labelList[np.argmax(listType)]
            
            #if sub_label:
                #restext = restext+"-"+sub_label
          
                                       
            
        return img,ocrtext, restext
    
if __name__ == '__main__':
    import time
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../ocr_models/Tesseract_API'))
    from TesseractAPI_SingleHandle_Class import TessAPI
    tess_api = TessAPI()
    tess_api.Tess_API_Init(lang = 'chi_new_stsong_jx',flag_digit = 0,psm = 6)#classify, ty_gray_ostu_0824  3
    tess_api_vert = TessAPI()
    tess_api_vert.Tess_API_Init(lang='chi_new_stsong_jx', flag_digit=0, psm=5)
    
    dirpath = r"../../test/test"
    filelist = os.listdir(dirpath)
    filelist = [os.path.join(dirpath,basename) for basename in filelist if basename.endswith('.JPG') or basename.endswith('.jpg')]
    #filelist = [r"../../test\1.JPG"]
    for imgPath in filelist:
        print(imgPath)
        #img = cv2.imdecode(np.fromfile(imgPath, dtype = np.uint8), -1)
        img = cv2.imread(imgPath)
        #cv2.imshow("img",img);cv2.waitKey(0)
        start = time.time()
        img,ocrtext, restext = ocrBillTitle(img, tessAPI=tess_api,tess_api_vert =tess_api_vert).do()
        print(restext)
        timeRange = time.time()-start
        print("time is {}".format(timeRange))
        
        #cv2.imshow("ocrORI",img);cv2.waitKey(0)
        # cv2.imwrite(imgPath[:-4]+".png", img)
    
    #imgPath = unicode(r"D:\01work\python\test\GZYH_OCR\test\test_shenjing\860575ad01f246aba6e3daa952a6f5ec.JPG".decode('utf-8'))
    ##imgPath = unicode(r'\\192.168.0.80\软一中银数据\zy_type_sample\支票\转账支票\1.JPG'.decode('utf-8'))
    
    ##os.path.join(a)
    #img = cv2.imdecode(np.fromfile(imgPath, dtype = np.uint8), -1)
    
    #start = time.time()
    #img,ocrtext, restext = ocrBillTitle(img, tess_api).do()
    #print(restext)
    #cv2.imshow("ocrORI",img);cv2.waitKey(0)
    ##print ocrtext, restext 
    ##print 'time:', time.time() - start
    
    
    #import xlsxwriter
    #from io import BytesIO  
    #import glob
    #import shutil
    ##import sys
    ##reload(sys) 
    ##sys.setdefaultencoding('utf-8')     
    
    #testPath = r"C:\testData\bank_ticket_entry_data\guizhou\dst_img"
    #filepaths = glob.glob(testPath+"/*.jpg")
    #cutPath = os.path.join(testPath,"cutImg_new_2")
    #if not os.path.exists(cutPath):
        #os.makedirs(cutPath)
    
    #writePath = r"C:\testData\bank_ticket_entry_data\guizhou\result_type_new" 
    #imgList = []
    #results = []
    #resList = []
    #writeExcel = os.path.join(testPath,"valueResult_new_2.xlsx")
    #for testFile in filepaths:
        #ori_img = cv2.imdecode(np.fromfile(testFile, dtype = np.uint8), -1)
        #img,ocrtext, restext = ocrBillTitle(ori_img, tess_api).do()
        #print(ocrtext)
        #print(restext)
        ##if img is not None:
        #if isinstance(img, np.ndarray):
            #cutimgPath = os.path.join(cutPath,os.path.basename(testFile))
            #cv2.imencode('.jpg', img)[1].tofile(cutimgPath)  
            #imgList.append(cutimgPath)
        #else:
            #imgList.append(None)
        #results.append(restext)
        #resList.append(ocrtext)
        #if restext == None:
            #tmpWritePath = os.path.join(writePath,"noType")
        #else:        
            #tmpWritePath = os.path.join(writePath,restext)
        #if not os.path.exists(tmpWritePath):
            #os.makedirs(tmpWritePath)
        #shutil.copy(testFile,os.path.join(tmpWritePath,unicode(os.path.basename(testFile),"gbk")))
    #tess_api.Tess_API_Delete()    
        

    ##imgList = []
    ##results = []
    ##writeExcel = os.path.join(testPath,"testvalue.xlsx")
    ##import glob
    ##for testFile in filepaths:
        ###print(unicode(testFile,"gbk"))
        ###ocrBillTitle_Ty(testFile, tess_api).getBottomRight_cornerImg()
        ##img,ocrtext, restext = ocrBillTitle(testFile, tess_api).do()
        ##if img is not None:
            ###print(img.shape)
            ##cutimgPath = os.path.join(cutPath,os.path.basename(testFile))
            ##cv2.imencode('.jpg', img)[1].tofile(cutimgPath)  
            ##imgList.append(cutimgPath)
        ##else:
            ##print(unicode(testFile,"gbk"))
            ##imgList.append(None)
        ##results.append(ocrtext)
    
    #workbook  = xlsxwriter.Workbook(writeExcel)
    #worksheet = workbook.add_worksheet()
    #f = workbook.add_format({'bold': True, 'bg_color': 'yellow'})
    #worksheet.write(0,0,"error_num",f)
    #worksheet.write(0,1,"total_num",f)
    #worksheet.write(0,2,"correct_rate",f)
    
    #worksheet.write(2, 0, 'name', f)
    #worksheet.write(2, 1, 'photo', f)
    #worksheet.write(2, 2, 'preResult', f)
    #worksheet.write(2, 3, 'typeResult', f)
    #worksheet.write(2, 4, 'isError',f)

    #worksheet.set_column(0, 1, 50)
    #errorNum = 0
    #row = 3
    #col = 0
    #totalNum = len(results)
    #for i in range(totalNum):
        #worksheet.set_row(row, 60)
        
        #filename = os.path.basename(filepaths[i])
        #worksheet.write(row,col,unicode(filename,"gbk"))
        #if imgList[i] is not None:
            #image_data = BytesIO(open(imgList[i], 'rb').read())
            #worksheet.insert_image(row,col+1,unicode(filename,"gbk"),{'image_data': image_data,'x_scale': 0.8, 'y_scale': 0.8, 'positioning': 2})        
        
        #worksheet.write(row,col+2,resList[i])
        #worksheet.write(row,col+3,results[i])
        
        #if results[i] in unicode(filename,"gbk"):
            #worksheet.write(row,col+4,True)
        #else:
            #worksheet.write(row,col+4,False)
            #errorNum = errorNum+1
        #row = row+1
    #worksheet.write(1,0,errorNum)
    #worksheet.write(1,1,totalNum)
    #worksheet.write(1,2,round((1-float(errorNum)/totalNum), 2))
    #workbook.close()