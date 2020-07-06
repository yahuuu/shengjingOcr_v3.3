# -*- coding: utf-8 -*-
import requests
import json
import os
import time
import base64
import chardet
testPath = "/home/alex/sjocr_v3.3/get_precision"
# testPath = r"../../get_precision"
# testPath = r"../../test_img_065"
# testPath = r"./test_img"
import glob
import cv2
import numpy as np
count = 0
sum_time = 0
from tomorrow import threads


@threads(10)#使用装饰器，这个函数异步执行
def  download(url,dataStr):
    #start = time.time()
    content = requests.post(url,json.dumps(dataStr)).content
    #time_range = time.time() - start
    resultTxt = json.loads(content)["type"]
    print(resultTxt)
    #print('time:', time_range)
    return time_range

def fileToBase64(img_path):    
    with open(img_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        # print(chardet.detect(encoded_image))
        image_string = encoded_image.decode('utf-8')
    return image_string
    # return encoded_image

#start1 = time.time()
#files = glob.glob(testPath+"/*.JPG")
#paramList = [{"param":fileToBase64(filename),"type":1} for filename in files]
#responses = [download('http://127.0.0.1:10001/getBillType',i) for i in paramList]
#time_range1 = time.time() - start1
#print('avgTime',time_range1/len(responses))

######print(responses)
def t1():
    for path in glob.glob(testPath+"/*/*[jpg, jpeg, JPEG, JPG]"):
        #data = {"path":unicode(path.decode('gbk'))}
        #dataStr = json.dumps(data)
        dataBase64 = fileToBase64(path)
        data = {"param":dataBase64,"type":1}

        ## 传递图片格式 会出现问题
        ##dataImg = cv2.imdecode(np.fromfile(path, dtype = np.uint8), -1)
        ##data = {"param":dataImg,"type":2}
        
        #data = {"param":"http://a4.att.hudong.com/21/09/01200000026352136359091694357.jpg","type":3}
        
        dataStr = json.dumps(data)
        # start = time.time()
        #content = requests.get('http://127.0.0.1:10001/getBillType?'+dataStr).content 
        content = requests.post('http://127.0.0.1:10002/getBillType',dataStr).content
        resultTxt = json.loads(content)["type"]
        
        #resultTxt = json.loads(content)[label]
        # time_range = time.time() - start
        # sum_time = sum_time+time_range
        count = count+1
        
        #print("{},result:{}".format(os.path.basename(path),content))
        print( resultTxt, path,  time_range)
    # print('avgTime',sum_time/count)#0.553

def t2():
    for root, dirs, files in os.walk(testPath):
        if not files: continue
        label =  os.path.basename(root) 
        # print(dir, files, filename)
        num = 0
        for idx, file in  enumerate(files):
            im_pth = os.path.join(root, file)
            dataBase64 = fileToBase64(im_pth)
            data = {"param":dataBase64,"type":1}
            dataStr = json.dumps(data)
            start = time.time()
            content = requests.post('http://127.0.0.1:10002/getBillType',dataStr).content
            resultTxt = json.loads(content)["type"]
            time_range = time.time() - start
            print( resultTxt,  im_pth,  time_range)
            if  label == resultTxt: num +=1
        print("precision:{}".format( (0.001+num)/(0.001+idx+1)) )

t2()








































