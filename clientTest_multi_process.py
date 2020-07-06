# -*- coding: utf-8 -*-
import requests
import json
import os,sys
import time
import base64
import glob
import cv2
import numpy as np
count = 0
sum_time = 0


from multiprocessing import Process, Queue,Manager

def loop(func_name,lock,queue,i,results):
    
    with lock:
        print('process  ' + str(i) + '  started...')
    
    while 1:
        filename = queue.get()
    
        if filename == -1:
            queue.put(-1)
            break
        
        time1 = time.time()
        dataStr = {"param":fileToBase64(filename),"type":1}
#        time1 = time.time()
        res = func_name(lock,'http://127.0.0.1:10001/getBillType',dataStr)
        time_cost = round(time.time() - time1,3)
        results.append((filename,res,time_cost))
        
        
def multi_process_func(func_name,filenames,max_process = 4,maxsize_num = 100000):
    queue_job = Queue(maxsize = maxsize_num + 1) ## 1: put(-1)
    for filename in filenames:
        queue_job.put(filename)
    queue_job.put(-1)
    
    manager = Manager()
    results = manager.list([])
    lock = manager.Lock()
    
    process_list = []
    for i in range(max_process):  ##max_process最大进程数, i:进程编号
        process_parse = Process(target = loop,args = (func_name,lock,queue_job,i,results))
        process_parse.start()
        process_list.append(process_parse)
    for process_parse in process_list:
        process_parse.join()
         
    return results


def download(lock,url,dataStr):
    content = requests.post(url,json.dumps(dataStr)).content
    resultdict = json.loads(content) #["type"]
    return resultdict

def fileToBase64(img_path):    
    with open(img_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        image_string = encoded_image.decode('utf-8')
    return image_string

def get_file_list(testPath,file_suffix = []):
    file_list = []
    for dir_path,_,basenames in os.walk(testPath):
        for basename in basenames:
            filename = os.path.join(dir_path,basename)
            file_list.append(filename)
    if len(file_list) and len(file_suffix):
        file_list = list(filter(lambda x:x[-4:] in file_suffix,file_list))
    return file_list


if __name__ == "__main__":
    import traceback
    if len(sys.argv) == 1:  #default      /testpath/*.jpg
        testPath = r"./test_img"
    else:
        testPath = sys.argv[1]
    
    print('testPath: {}'.format(os.path.abspath(testPath)))
    
        
    filenames = get_file_list(testPath,['.JPG'])

    time_ = time.time()
    
    results = multi_process_func(download,filenames,max_process = 4,maxsize_num = 100000)
    time_multi_cost = time.time() - time_
    
    
#    results = sorted(results,key=lambda x:x[0])
    
    sum_time = []
    for filename,res,time_cost in results:
        basename  = os.path.basename(filename)
        print('{} res is {}'.format(basename,res['type']))
                
        sum_time.append(time_cost)
    
    avgTime = np.mean(sum_time)
    print('avgTime per image one-process: {}'.format(round(avgTime,3)))
    
    print('avgTime per image multi-process: {}'.format(round(time_multi_cost/len(filenames),3)))
    
    
    
    
    
    
    
    
    
    
    
