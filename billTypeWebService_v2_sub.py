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
    from operationConfig import MyConf,Writepid
    from cnn_interface_sj.ApplicationFormClassification.interface import model, return_result_dict, generate
    from cnn_interface_sj.ApplicationFormClassification.interface import pred
    from templateMatch.TemplateMatch import ModelMatchInter
except:
    from sjocr.ocr_models.Tesseract_API.TesseractAPI_SingleHandle_Class import TessAPI
    from sjocr.bankBillTypeOCR.title_Type.billTitleOCRInterface import billType
    from sjocr.logger import logger_Info
    from sjocr.operationConfig import MyConf,Writepid
    from sjocr.cnn_interface_sj.ApplicationFormClassification.interface import model, return_result_dict, generate
    from sjocr.cnn_interface_sj.ApplicationFormClassification.interface import pred
    from sjocr.templateMatch.TemplateMatch import ModelMatchInter

import base64
import cv2
import urllib.request
import json
import numpy as np
# from operationConfig import MyConf,Writepid
# from logger import logger_Info
import traceback
import torch

tess_api = None
tess_api_vert = None
runlog = None
modelImgList = []
os.environ["CUDA_VISIBLE_DEVICES"] = "0" if torch.cuda.is_available() else "-1"
id_rt_value = {"IDCardBack" , "IDCardFront"}

model_match_inter = ModelMatchInter()

typeList = {
    u"结算业务申请书":           u"013",
    u"结算业务申请书-第一联":u"013",
    u"结算业务申请书-第二联":u"013",
    u"结算业务申请书-第三联":u"013",
    u"结算业务申请书（无号码）":           u"528",
    u"结算业务申请书（无号码）-第一联":u"528",
    u"结算业务申请书（无号码）-第二联":u"528",
    u"结算业务申请书（无号码）-第三联":u"528",
    u"进账单":   u"501",
    u"转账支票":u"001",
    u"上海贷记凭证大联（2、3联）":u"011",
    u"上海贷记凭证小联（1、4联）":u"011b",
    u"特种转账传票":u"520",
    u"托收凭证":       u"526",
    u"银行承兑汇票":u"008",
    u"商业承兑汇票":u"010",
    u"普通支票":u"002",
    u"银行本票":u"012",
    u"通用凭证":u"201",
    u"None":      u"None",

    u"IDCardBack":u"016",
    u"IDCardFront":u"015",
    u"现金缴款单":u"510",
    u"支款凭证":   u"701",
    u"zhczhqtzhchxcd":u"066",
    u"zhczhqdqchxcd":u"065",

    u"通知储蓄存单":u"069",
    u"单位定期存单":u"016",
    u"现金支票":u"003",
    u"单位定期存款开户证实书":u"014",
    u"单位结构性存款开户证实书":u"088",
    u"盛京银行大额存单申请书":u"103",
    u"委托付款授权确认书":u"055",
    u"单位银行结算账户短信通知服务申请书":u"901",
    u"批量业务申请单":u"902",
    u"盛京银行个人结构性存款产品协议书":u"101",
    u"盛京银行开立资信证明申请书":u"093",
    u"预制卡":u"61",
    u"资信证明书（正本）":u"None"
}


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
        type_result = ""
        return_result = {}
        try:
            data = json.loads(dataStr)
            param = data["param"]
            paramType = data["type"]
            return_result["url"] = param
            return_result["type"] = u"None"
            if paramType == 1:
                img = base64ToImg(param)
            elif paramType == 2:
                img = param
            elif paramType == 3:
                img = getImgByUrl(param)
            else:  # para error
                return_result["type"] = "para error: check type"
                return json.dumps(return_result)

            type_result = pred(img)
            # print("->cnn_result")
            if type_result not in id_rt_value:
                type_result = billType(img, tess_api, tess_api_vert, modelImgList)
            if type_result is "None":
                type_result = model_match_inter.get_class(img)

                # print("->tess_result")
            
            # type_result = billType(img, tess_api, tess_api_vert, modelImgList) #single tess

            # 根据识别结果返回 类别代码
            for item in typeList.keys():
                if item == type_result:
                    type_result = typeList[item]
                    break
                
            #print(type_result)
            return_result["type"] = type_result
            #runlog.info(type_result)
        except:
            runlog.error("运行失败: " + str(dataStr))
            runlog.error(traceback.format_exc())
            #print(e)

        return json.dumps(return_result)

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

def startWebService(server_port):
    server = WebServerApplication(server_port)
    server.process()


if __name__ == "__main__":
    # 配置文件写入进程号
    configPath = "./paramConfig.conf"
    cf = MyConf()
    cf.read(configPath)      
    currentPid = os.getpid()
    Writepid(configPath,cf,currentPid) 
    
    tess_api = TessAPI()
    tess_api.Tess_API_Init(lang = 'chi_new_stsong_jx',flag_digit = 0,psm = 6)
    tess_api_vert = TessAPI()
    tess_api_vert.Tess_API_Init(lang='chi_new_stsong_jx', flag_digit=0, psm=5)
    modelPathList = [os.path.join('./tmpl_model',itemPath) for itemPath in os.listdir(r'./tmpl_model') if itemPath.endswith(".png")]
    for m_imagePath in modelPathList:
        type = os.path.basename(m_imagePath).replace(".png","")
        m_image = cv2.imread(m_imagePath,0)
        modelImgList.append([type,m_image])

    server_port = "10002"
    
    #定义服务端口
    if len(sys.argv)>1:
        server_port = sys.argv[1]
    
    logfilename = "./runLog_"+server_port+".log"
    runlog = logger_Info(logIndex="debug",logPath=logfilename)
    
    server = WebServerApplication(str(server_port))
    server.process()