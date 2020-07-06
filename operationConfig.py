#coding:utf-8

import configparser
from queue import Queue
import sys

class MyConf(configparser.ConfigParser):
    def __init__(self,defaults=None):
        configparser.ConfigParser.__init__(self,defaults=None)
    def write(self, fp):
        if self._defaults:
            fp.write("[%s]\n" % DEFAULTSECT)
            for (key, value) in self._defaults.items():
                fp.write("%s = %s\n" % (key, str(value).replace('\n', '\n\t')))
            fp.write("\n")
        for section in self._sections:
            fp.write("[%s]\n" % section)
            for (key, value) in self._sections[section].items():
                if key == "__name__":
                    continue
                if (value is not None) or (self._optcre == self.OPTCRE):
                    key = "=".join((key, str(value).replace('\n', '\n\t')))
                fp.write("%s\n" % (key))
            fp.write("\n")
                 
def Writepid(configPath,cf,currentPid):
    pid = cf.get("webserviceParam", "pid")
    if len(pid)>0:
        pid =pid+","+str(currentPid)
    else:
        pid = str(currentPid)
    cf.set("webserviceParam", "pid", pid)
    cf.write(open(configPath,'w'))
    
def getSubPortListAndWritePort(configPath,cf,currentPid):
    subportlistStr = cf.get("webserviceParam", "subportlist")
    queue = Queue(maxsize=10)
    for portNum in subportlistStr.split(","):
        queue.put(portNum)    
    Writepid(configPath,cf, currentPid)
    return queue

def clearPid(configPath,cf):
    cf.set("webserviceParam", "pid", "")
    cf.write(open(configPath,'w'))
if __name__ == "__main__":
    cf = MyConf()
    configPath = "./paramConfig.conf"
    cf.read(configPath) 
    
    if len(sys.argv)>1:
        operation = sys.argv[1]
        if operation=="clrearPid":
            clearPid(configPath,cf)
    