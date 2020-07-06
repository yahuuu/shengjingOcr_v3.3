#-*- coding:utf-8 -*-
'''
生成两种形式log文件：
   1. bug产生的文件
   2. 运行的文件
'''
import logging
def logger_Info(logIndex="info",logPath='./log/run.log'):
    # log初始化
    logLevelList = {"info":logging.INFO,"debug":logging.DEBUG}
    logLevel = logLevelList[logIndex]
    logger = logging.getLogger("billType service")
    logger.setLevel(level = logLevel)
    if not logger.handlers:
        handler = logging.FileHandler(logPath)
        handler.setLevel(logLevel)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
