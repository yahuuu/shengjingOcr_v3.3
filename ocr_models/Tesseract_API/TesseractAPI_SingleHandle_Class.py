# -*- coding: utf-8 -*-
"""
Created on Tue Jun 05 09:25:04 2018

@author: rcx
"""

import ctypes
import os
import sys
import cv2
import platform 


'''
##说明：  1. 将DLL 的路径 添加到 os.environ['PATH'] 
          2. Error 126 : 缺少其他依赖库
          3. TESSDATA_PREFIX tesseract.exe 的安装文件夹
          4. 需要重新加载 lang 则花大量时间初始化  (所以得合并图，不然重复初始化加载不同traindata耗时)
          5. TESSDATA_PREFIX 需要str 类型 ，os.path 是unicode
          6. OEM =3 时，设置TessBaseAPISetVariable 也会有影响
          7. 存在多实例时，初始化要在函数里面初始化
                可能因为： 多handle在 函数里初始化时，会经过函数的包装，可以避免内存泄露
          8. 多实例应该在导入此包时 都初始化好，其他py直接调用这里的识别方法，而不能其他函数调用这里的初始化，
                handle的传递，tesseract的内存会出错 
          9. !!! 'chi_sim + test05' 会报内存错误，一定要改成'chi_sim+test05'
          10. 未解决问题：初次识别代码对，后面错， LSTM 没找着
'''

#print('Runing Tesseract_API_64_32 Init()')

#TESSDATA_PREFIX = os.environ.get('TESSDATA_PREFIX')
#if not TESSDATA_PREFIX:
#TESSDATA_PREFIX =  "C:/Program Files (x86)/Tesseract-OCR/tessdata"
class TessAPI:
    def __init__(self):
        libdirpath_win = os.path.dirname(os.path.abspath(__file__)) #ocr_models\Tesseract_API
        #self.TESSDATA_PREFIX = libdirpath_win
        self.TESSDATA_PREFIX = os.path.join(os.path.dirname(libdirpath_win),'tessdata')
        self.TESSDATA_PREFIX =  bytes(self.TESSDATA_PREFIX,encoding='utf-8')
        
        if sys.platform =="win32":   ##操作系统 win                 
            if '64 bit' in sys.version  or '64' in platform.architecture()[0]:  ##64位python
                self.flag_64bit = 1
                libpath_win_4 = libdirpath_win + '/64bit/TesseractDLL4/'  ##libpath_win:  dll所在的文件夹
                DLL_PATH_4 = libpath_win_4+ 'tesseract40.dll'
                os.environ["PATH"] += os.pathsep + libpath_win_4      
                
                DLL_PATH_3 = '' ## 使得 下方打印 不报错；为64位python 预留tesseract3.0 接口
                      
            else:  ##32位python
                self.flag_64bit = 0 
                libpath_win_4 = libdirpath_win + '/32bit/TesseractDLL4/'
                DLL_PATH_4 = libpath_win_4 + 'tesseract40.dll'
                os.environ["PATH"] += os.pathsep + libpath_win_4
                
                libpath_win_3 = libdirpath_win + '/32bit/TesseractDLL3/'
                DLL_PATH_3 = libpath_win_3 + 'libtesseract-3.dll'                           
                os.environ["PATH"] += os.pathsep + libpath_win_3     
                    
            try:
                self.tesseract = ctypes.cdll.LoadLibrary(DLL_PATH_4)
            except:
                try:
                    self.tesseract = ctypes.cdll.LoadLibrary(DLL_PATH_3)
                except:
                    print('DLL_PATH_3: {}'.format(DLL_PATH_3))
                    print('DLL_PATH_4: {}'.format(DLL_PATH_4))           
            
        else: ##OS: linux
            if '64 bit' in sys.version  or '64' in platform.architecture()[0]:
                self.flag_64bit = 1
            else:
                self.flag_64bit = 0
                
            libpath = os.path.dirname(os.path.abspath(__file__)) 
            libname = '/linux/libtesseract.so.4.0.0'       #libtesseract.so.3.0.4      libtesseract.so.4.0.0
            DLL_PATH = libpath+ libname
            
            self.tesseract = ctypes.cdll.LoadLibrary(DLL_PATH)         
            
        self.c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
        if self.flag_64bit:    
            self.tesseract.TessBaseAPICreate.restype = ctypes.c_uint64
            self.tesseract.TessBaseAPIGetUTF8Text.restype = ctypes.c_uint64
            self.tesseract.TessBaseAPISetImage.restype = ctypes.c_uint64
            #tesseract.TessBaseAPISetImage.argtypes = [ctypes.c_uint64,c_ubyte_p,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64]
        #    tesseract.TessBaseAPIRecognize.restype = ctypes.c_uint64 ##成功返回0，结果存在内部数据
            
            self.handle = self.tesseract.TessBaseAPICreate()
            self.handle = ctypes.c_uint64(self.handle)
        else:
            self.handle = self.tesseract.TessBaseAPICreate()
            
        ##其他self变量
        self.lang = None
        self.digit = 0
        self.psm = 3

#Alphabet_ = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#Num_ = '0123456789'

#digit_dict = {0:3,1:0}  ##flag_digt =0 时 oem =3, =1 时 oem = 0
    def Tess_API_Init(self,lang = 'chi_sim',flag_digit = 0,psm = 3):   
        lang = bytes(lang,encoding='utf-8')
    
        #rc = tesseract.TessBaseAPIInit3(ctypes.c_uint64(handle), TESSDATA_PREFIX, lang)

        if flag_digit:
            self.tesseract.TessBaseAPIInit2(self.handle, self.TESSDATA_PREFIX, lang,0)  ##0 表示初始化成功
                                                                            ##+oem : 1，2，3比0 快
                                                                               # 0 Original Tesseract only
                                                                               # 1  Neural nets LSTM only
                                                                               # 2 Tesseract + LSTM
                                                                               # 3 Default, based on what is available                                                                          
            self.tesseract.TessBaseAPISetPageSegMode(self.handle,psm) ##--psm 3
            self.tesseract.TessBaseAPISetVariable(self.handle,'tessedit_char_whitelist','0123456789')
    
    #    tesseract.TessBaseAPISetVariable(handle,'tessedit_char_whitelist',Alphabet_+Num_+'<')
        else:
            self.tesseract.TessBaseAPIInit2(self.handle, self.TESSDATA_PREFIX, lang,3) #3比0快10倍.
            self.tesseract.TessBaseAPISetPageSegMode(self.handle,psm) ##--psm 3
        
        self.lang = lang
        self.digit = flag_digit
        self.psm = psm

    def Tess_API_OCR_filename(self,filename):
        self.tesseract.TessBaseAPIProcessPages(
            self.handle, filename, None, 0, None)
        text_out = self.tesseract.TessBaseAPIGetUTF8Text(self.handle)
        return ctypes.string_at(text_out)

    def Tess_API_OCR_Image(self,image):
        row,col = image.shape

    #    c_ubyte_p = ctypes.POINTER(ctypes.c_ubyte)
    #    tesseract.TessBaseAPISetImage(handle,image.ctypes.data_as(c_ubyte_p),ctypes.c_uint64(col),ctypes.c_uint64(row),ctypes.c_uint64(1),ctypes.c_uint64(col))
    
        self.tesseract.TessBaseAPISetImage(self.handle,image.ctypes.data_as(self.c_ubyte_p),col,row,1,col)
        self.tesseract.TessBaseAPIRecognize(self.handle,None)

        
        text_out = self.tesseract.TessBaseAPIGetUTF8Text(self.handle)
    #    text_out = tesseract.TessBaseAPIGetUNLVText(handle)    
        text_out = ctypes.string_at(text_out)

        return text_out

    def Tess_API_Delete(self):
        self.tesseract.TessBaseAPIDelete(self.handle)  ##handle 为None也不报错
        
        


if __name__ == '__main__':
    import time
    image_file_path = './1_300_20.png'
    
    image = cv2.imread(image_file_path,-1)
    
    tess_api = TessAPI();print('after Init()')        #注意测试时，lang 应改为识别出问题的tess_api 的语言包。 因为 chi_sim 往往没有问题。
    #tess_api.Tess_API_Init(lang = bytes('chi_sim',encoding='utf-8'),flag_digit = 0,psm = 7)
    tess_api.Tess_API_Init(lang = 'chi_sim',flag_digit = 0,psm = 7) 

    
    try:
        result = tess_api.Tess_API_OCR_Image(image);result = str(result,encoding='utf-8')
    except:
        pass

    time1 = time.time()
    result = tess_api.Tess_API_OCR_Image(image);result = str(result,encoding='utf-8')
    time_cost = round(time.time() - time1,3)
    # if isinstance(result,str):
        # result = unicode(result,'utf-8')
        
    print('ocr result: ')
    print (result)
    print('time_cost: {}'.format(time_cost))
    
    tess_api.Tess_API_Delete()
    
    
    
    
    
    
    
    