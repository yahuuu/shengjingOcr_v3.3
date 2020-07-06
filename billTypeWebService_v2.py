#coding:utf-8
import os
import sys
import tornado
import tornado.ioloop
import tornado.web
from concurrent.futures import ThreadPoolExecutor
try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), './bankBillTypeOCR/title_Type'))
    from billTitleOCRInterface import billType
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), './ocr_models/Tesseract_API'))
    from TesseractAPI_SingleHandle_Class import TessAPI
    from logger import logger_Info
    from operationConfig import MyConf,getSubPortListAndWritePort
except:
    from sjocr.ocr_models.Tesseract_API.TesseractAPI_SingleHandle_Class import TessAPI
    from sjocr.bankBillTypeOCR.title_Type.billTitleOCRInterface import billType
    from sjocr.operationConfig import MyConf,Writepid
    from sjocr.operationConfig import getSubPortListAndWritePort
    from sjocr.logger import logger_Info

import base64
import cv2
import numpy as np
import urllib.request
import json
import requests
# from operationConfig import MyConf,getSubPortListAndWritePort
# from logger import logger_Info
import traceback
import torch

port_queue = None
runlog = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else "-1"

def base64ToImg(image_string):
    img_data = base64.b64decode(image_string)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

def  getImgByUrl(imgSrc):
    resp = urllib.request.urlopen(imgSrc)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

class Executor(ThreadPoolExecutor):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not getattr(cls, '_instance', None):
            cls._instance = ThreadPoolExecutor(max_workers=10)
        return cls._instance

class BillTypeHandler(tornado.web.RequestHandler):
    executor = Executor()

        
    @tornado.web.asynchronous  # 异步处理
    @tornado.gen.coroutine  # 使用协程调度
    def post(self):
        """ get 接口封装 """

        # 可以同时获取POST和GET请求参数
        dataStr = self.request.body
        result = yield self._process(dataStr)
        self.write(result)    

    @tornado.concurrent.run_on_executor  # 增加并发量
    def _process(self, dataStr):
        # 此处执行具体的任务 
        portStr = port_queue.get()
        #print(portStr)
        try:
            content = requests.post('http://127.0.0.1:port_num/getBillType'.replace("port_num",str(portStr)),dataStr).content
        except:
            runlog.error("运行失败：端口号为{} ;传递信息为：{}".format(str(port_num),str(dataStr)))
            runlog.error(traceback.format_exc())            
        port_queue.put(portStr)
        return content        

class WebServerApplication(object):
    def __init__(self, port):
        self.port = port
        #self.settings = {'debug': False, 'autoreload':False}
        self.settings = {'debug': False}

    def make_app(self):
        """ 构建Handler
        (): 一个括号内为一个Handler
        """

        return tornado.web.Application([
            (r"/getBillType?", BillTypeHandler)
            ], ** self.settings)

    def process(self):
        """ 构建app, 监听post, 启动服务 """

        app = self.make_app()            
        app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    configPath = "./paramConfig.conf"
    cf = MyConf()#ConfigParser.ConfigParser()
    cf.read(configPath)     
    currentPid = os.getpid()
    port_queue = getSubPortListAndWritePort(configPath,cf,currentPid)
    # 定义服务端口
    server_port = "10001"

    if len(sys.argv)>1:
        server_port = sys.argv[1]
        
    logfilename = "./runLog_"+server_port+".log"
    runlog = logger_Info(logIndex="debug",logPath=logfilename)
    
    server = WebServerApplication(server_port)
    server.process()
