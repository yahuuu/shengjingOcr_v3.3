#!/usr/bin/python3  
# -*- coding:utf-8 -*-
'''
@attention: 
@author: wsd
'''
import os
import cv2
import numpy as np
import shutil
import math
import re
import codecs

def getCSVLines(csvname):
    '''
    @attention: 获取csv文件所有行
    @param csvname: 文件
    '''    
    csvFile = codecs.open(csvname, mode = 'r', encoding = 'utf-8')
    lines = csvFile.readlines()
    temps = []
    for line in lines:
        temps.append([x.strip() for x in line.split(',')])
    csvFile.close()
    return temps
    # csvFile.close()
    
def pltShow(imgs,tis, cmap_ = 'gray', isShow = True, fileName = '' ):
    '''
    @attention: 使用plt显示图片
    '''
    from matplotlib import pyplot as plt
    lenImg = math.ceil(len(imgs) / 2)
    for pos,im  in enumerate(imgs):
        plt.subplot(2,lenImg ,pos+1), plt.imshow(im, cmap = cmap_), plt.title(tis[pos])
    #保存图片
    if fileName != '': plt.savefig("filename.jpg")
    #显示图片
    if isShow: plt.show()

def justTYtitle(res):
    '''
    @attention: 判断字符串res属于那种标题
    @return: 返回对应标题的数字，如果为10则不匹配TY标题
    '''  
    if isinstance(res, str):
        res = res.decode('utf-8')
    #处理文字，去除空格和回车 
    res = handleText(res)
    res = re.sub(u'\.', u'', res)
    res = re.sub(u'(\(\))|(O)|(\(1)|(1\))|(\))|(\()', u'0', res)
    res = re.sub(u'(\))|(\()', u'0', res)
    res = re.sub(u'(T7)|(1Y)|(0Y)', u'TY', res)
#     print  'res',res       
#     #如果有T没有Y，把T后面的字符替换成Y
#     if res.count('T') == 1 and res.count('Y') == 0:
#         ind = res.index('T')
#         if ind + 3 < len(res):
#             res = res[0:ind + 1] + 'Y' + res[ind+2:]
    value = u'None'
    #把Y后面非0的字符删除 ,以及把Y后面的T替换成0
    if 'Y' in res:
        ind = res.index('Y')
        last = res[ind:].replace('T', '0')
        res = res[0:ind] + last
        if ind + 2 < len(res) and res[ind+1] != '0':
            res = res[0:ind+1] + res[ind+2:]
    else: 
        return res, value 
    #print 'res',res
    #通过正则表达式判断，其中0为0个或者多个
    if len(re.findall('TY0*5', res)) != 0:      value = u'TY05'
    elif len(re.findall('TY0*4', res)) != 0:    value = u'TY04'
    elif len(re.findall('TY0*2', res)) != 0:    value = u'TY02'
    
    return res, value 

def handleZYtitle(res):
    '''
    @attention: 对识别出来的中银文字进行简单的增删改处理，
    @return: 返回修改后的标题
    ''' 
    if isinstance(res, str):
        res = res.decode('utf-8')
        
    #替换字母、数字(除了91)和.():为空
    res = re.sub(u'[a-zA-Z.()：:一1]', u'', res)
    res = re.sub(u'帐', u'账', res) #把帐改为账
    #易错识别替换
    res = re.sub(u'明.{0,1}临', u'999临', res)
    res = re.sub(u'临日', u'临时', res)
    res = re.sub(u'转种转账', u'特种转账', res)
    res = re.sub(u'(特账)|(存专账)|(传账)', u'转账', res)
    res = re.sub(u'1专票', u'传票', res)
    res = re.sub(u'财正支', u'财政', res)  
    res = re.sub(u'款挂条', u'款凭条', res) 
    res = re.sub(u'出人', u'出入', res) 
    res = re.sub(u'收人', u'收入', res)
    res = re.sub(u'请单', u'清单', res)
    res = re.sub(u'申清', u'申请', res)
    res = re.sub(u'错财冲', u'错账冲', res)
    res = re.sub(u'中正', u'冲正', res)
    res = re.sub(u'个人客支', u'个人客户', res)
    res = re.sub(u'中回银行', u'中国银行', res)
    res = re.sub(u'进账申', u'进账单', res)
    res = re.sub(u'(现个)|(人正人)|(入正人)|(人支人)|(入支人)|(门人.{0,1}人)|(门入.{0,1}人)|(金正人)|(正门.{0,1}人)', u'现金', res)
    res = re.sub(u'凭正', u'凭证', res)
    res = re.sub(u'缴款申', u'缴款单', res)
    res = re.sub(u'金缴结', u'金缴款', res)
    res = re.sub(u'金缴单算', u'金缴款', res)
    res = re.sub(u'金缴金汇', u'金缴款', res)
    res = re.sub(u'(现合古票)|(现合专票)|(明金支票)|(现金支.{0,3}票)', u'现金支票', res)
    #res = re.sub(u'(现转支票)|(支存金支票)', u'转账支票', res)
    #res = re.sub(u'(行重金账)', u'行转账', res)
    #财政的更换 
    res = re.sub(u'(授中交)|(授中支)|(接月拨)|(接税装)', u'授权', res) 
    res = re.sub(u'(财正文)|(财正交)|(财正金)', u'财政', res) 
    res = re.sub(u'(政委要)|(政委更)|(财正金)|(政委委)', u'政授', res) 
    res = re.sub(u'(水文支)|(明文支)|(个文支)|(文支)', u'权支', res) 
    res = re.sub(u'(财.{0,1}挂.{1,3}直重)|(财现报缴接)', u'财政资金', res) 
    #res = re.sub(u'(凭账)', u'凭证', res)
    return res

def getTypes(filePath):
    '''
    @attention: 读取配置信息，获取类型对应表
    @param filePath: 配置文件路径 
    @return: 返回类型对应dist
    '''  
    #读取文件获取类型对应表
    #filePath = os.path.dirname(__file__) + '/billType.cfg'
    doc = codecs.open(filePath, mode = 'r', encoding = 'utf-8')
    typeStr =  ''.join(doc.readlines())
    doc.close()
    
    types = []
    typeStr = typeStr.split('\n')
    for ai in typeStr:
        if len(ai.strip()) == 0:    continue
        ls = ai.split('\t')
        key = ls[0].strip()
        value = ls[1].strip()
        #print key, '\t', value
        types.append([len(key), key, value])
    types.sort(key=lambda types:types[0], reverse=True)
    
    return types

def justZyTitle(res, types):
    '''
    @attention: 判断字符串res属于那种标题
    @param types: 判断类型列表
    @return: 返回对应标题
    ''' 
    res = handleZYtitle(res)
    tempRes = re.sub(u'[0-9]', u'', res)
    #print res
    #采用精确查询
    for item in types:
        key, value = item[1], item[2]
        if key in res:      
            if u'结算业务' in res:      value = u'结算业务申请书'
            return value
    #对财政类的票据进行处理
    czzfu = u'(财.{3,6}付.{0,1}凭.{0,1}证)|(政.{2,5}付.{0,1}凭.{0,1}证)|(财.{3,6}支.{0,1}付.{0,1}凭)|(政.{2,5}支.{0,1}付.{0,1}凭.{0,1}证)'
    rl = re.findall(czzfu, res)
    #计算概率，选取最大的概率值
    maxRatio, proKey, proValue = 0.0, u'None', u'None'
    #如果精确查询没有找到，采用模糊查询
    for item in types:
        key, value = item[1], item[2]
        #print key,'\t', value
        num = 0.0
        tstr = res
        if len(rl)>0:       tstr = u''.join(rl[0])
        #如果字数大于6个,选取包含字数概率大于0.7且选最大的概率
        akey = u''#用于限定该类型被选中的字的距离之间不能太远
        if len(key) > 6:
            for pk, ki in enumerate(key):
                if ki in tstr:   
                    num += 1
                    ind = tstr.index(ki)
                    tstr = tstr[ind:]
                    #if u'财政' in key:    print key, tstr
                    akey+=ki
                akey += u'.{0,3}'
            ratio = num / len(key)
            if ratio > 0.7 and ratio > maxRatio and len(re.findall(akey, res)) > 0:    
                maxRatio, proKey, proValue = ratio, key, value
        #如果字数为5个或者6个的情况，只能有1个字是错的
        if len(key) == 5 or len(key) == 6:
            #对7字以上的的key中进行处理
            if maxRatio > 0.7: 
                if u'财政' in proKey and u'财政资' in res:      proValue = u'财政资金支付凭证'
                if proKey == u'财政资金支付凭证':
                    if u'政性' in res or u'性资' in res:      proValue = u'财政性资金支付凭证'
                    if u'授权支付' in res  :      proValue = u'财政授权支付凭证'
                if proValue == u'境内-外汇款申请书' and (u'客户' in res or u'人民' in res):   proValue = u'个人客户境内汇款申请书'
                if proValue == u'特种转账传票' and u'进账' in res: proValue = u'进账单'
                
                return proValue
            if len(re.findall(u'(政直接支)|(直接支付)|(财政直.{0,3}接支)', tempRes)) > 0:     return u'财政直接支付凭证'
            if len(re.findall(u'(凭.{0,4}收账通知)|(凭证.{0,2}收账通)', tempRes)) > 0:     return u'财政资金支付凭证'
            if len(re.findall(u'(授.{0,1}权.{0,4}凭证)|(授.{0,1}权.{0,4}支付凭)|(财政.{0,1}授.{0,1}权)|(政授.{0,1}权支)', tempRes)) > 0:     return u'财政授权支付凭证'
            if len(re.findall(u'(99.{0,6}欠.{0,1}凭)|(99.{0,6}存.{0,1}欠)|(^99.{0,2}临.{0,2}时)', res)) > 0:     return u'9991临时存欠凭证'
            if len(re.findall(u'(个.{0,1}开户及综)|(个人开户及)|(个人开.{0,4}综合服.{0,2}申)|(人开户及综)',  tempRes)) > 0:     return u'个人开户及综合服务申请表'
            if len(re.findall(u'(^结算业务)|(算业务申)|(^结算.{0,1}务申)', re.sub(u'[0-9]', u'', tempRes))) > 0:     return u'结算业务申请书'
            if len(re.findall(u'(业务种类行.{0,4}款)|(业务种类行.{0,4}汇)|(业务种.{0,3}内)',  tempRes)) > 0:     return u'结算业务申请书'
            #用正则表达式进行判断
            for i in range(len(key)):
                keys = [x for x in key]
                keys[i] = u'P'
                remain = 'P'.join(keys)
                pattern = remain.replace('P', u'.{0,1}')
                #print i, remain, '\t',pattern
                #正则表达式， 字符间存在0到1个干扰字符
                if len(re.findall(pattern, tempRes)) > 0:     return value
            continue
        
        #如果字数为4个的情况    
        if len(key) == 4:
            #对5-6的key中现金缴款单和特种转账传票进行处理
            if len(re.findall(u'(兑换水)|(外汇.{0,1}兑)|(汇兑换水)|(汇兑.{0,3}水单)|(换水单)|(兑.{1,3}水单)|(汇兑换.{1,3}单)', tempRes)) > 0:      return u'外汇兑换水单'
            if len(re.findall(u'(人期存单)', tempRes)) > 0 or (len(tempRes)<6 and u'人期单' in  tempRes):      return u'个人定期存单'
            if len(re.findall(u'(现.{1,3}缴款)|(现金.{1,4}缴款)|(金.{0,2}缴款单)|(人现缴款)', tempRes)) > 0:      return u'现金缴款单'
            if len(re.findall(u'(款通知.{0,5}转账)|(种转账传)|(转账传票)|(特.{1,2}转账传)', tempRes)) > 0:      return u'特种转账传票'
            if len(re.findall(u'(^外汇兑换)|(^外汇单换)|(^外汇兑挂支)', tempRes)) > 0:      return u'外汇兑换水单'
            if len(re.findall(u'(银行单位存)|(^单位存.{0,1}算.{0,3}人凭条)|(单位存.{0,3}凭.{0,3}期)', tempRes)) > 0:      return u'单位存款凭条'
            if len(re.findall(u'(进账.{0,3}方凭)|(进申单.{0,2}方凭)|(账单.{0,2}方凭)', tempRes)) > 0:      return u'银行进账单'
            #用正则表达式进行判断
            pattern = u'.{0,2}'.join(key)
            #print pattern, u''.join(re.findall(pattern, res))
            if len(re.findall(pattern, tempRes)) > 0:      return value
            continue

        #如果类型为进账单
        if key == u'进账单':
            #对4字的key中现金支票和转账支票进行处理
            xjzpPa = u'(^.{0,10}现金.{0,2}支.{0,2}票)|(银行.{0,3}金支)|(^.{0,10}明正.{0,1}支票)'
            xjzpPa += u'|(^.{0,10}现.{1,2}支票)|(^.{0,10}现金支.{1,2})|(^.{1,10}金支票)|(^.{0,10}现金.{1,2}票)'
            zzhpPa = u'(^.{0,10}转.{1,2}支票)|(^.{0,10}转账支.{1,2})|(^.{0,10}账支票)|(^.{0,10}转账.{0,2}票)'
            if len(re.findall(xjzpPa, res)) > 0:
                if len(re.findall(u'(方凭证)|(现金缴.{0,1}票)', tempRes)) == 0:    return u'现金支票'
            if len(re.findall(zzhpPa, res)) > 0:
                if len(re.findall(u'(方凭证)|(^.{0,1}特)', tempRes)) == 0:    return u'转账支票'
            #对4字的key中单位存款和单位取款进行处理
            if len(re.findall(u'(单位存.{1,4}凭)|(单位行款凭条)|(单付合自款凭)', tempRes)) > 0 :   return u'单位存款凭条'
            if len(re.findall(u'(单位取.{1,4}凭)|(单位取.{1,4}条)', tempRes)) > 0 :   return u'单位取款凭条'   
                
            #用正则表达式进行判断
            if len(re.findall(u'(进.{0,1}账单)|(进账.{0,1}单)', tempRes)) > 0:   return value
            continue
    
    #疑似存折的情况
    if len(re.findall(u'\d{18}', res)) > 0 and res.count('1') < 8 and res.count('0') < 8:     return u'no'  
    
    return u'None'

def handleText(text):
    '''
    @attention: 处理文字，去除空格和回车
    '''    
    text = text.strip()
    text = [x for x in text if x != '\n']
    text = [x for x in text if x != ' ' ]
    return ''.join(text)

def ocrIdCard(img, langName = 'eng' , psm = 7):
    '''
    @attention: 识别文字
    @param image: 二值化的图像
    '''
    import pytesseract
    from PIL import Image
    image = Image.fromarray(img)
    #pytesseract 文字识别 
    tessdata_dir_config = '--tessdata-dir "' + os.environ['TESSDATA_PREFIX'] + '" --psm ' + str(psm)
    result = pytesseract.image_to_string(image, lang=langName, config = tessdata_dir_config)
    return result

def extract_peek_ranges_from_array(array_vals, drange):
    '''
    @attention: 从数组中获取峰值的间隔坐标
    @param drange: 最小的间隔
    '''    
    minVal = np.min(array_vals)
    maxVal = np.max(array_vals)
    dval = int((maxVal - minVal) * 0.01)
    #print('drange, dval, minVal + dval', drange, dval, minVal + dval)
    
    tempi, tempVal  = None, minVal 
    peek_ranges = []
    for i, val in enumerate(array_vals):
        
        if tempi is None and val > minVal + dval:
            tempi = i
        elif tempi is not None and val < minVal + dval:
            if i - tempi >= drange:
                peek_ranges.append((tempi, i))
                tempi = None
            else:
                tempi = i
                
        elif tempi is not None and i == len(array_vals)-1:
            if i - tempi >= drange:
                peek_ranges.append((tempi, i))
        
    plt.plot(array_vals)
    #plt.show()
    
    return peek_ranges

def drawCountourAndShow(img, contours, windowName = 'img', color = 255 ):
    '''
    @attention: 在图片上画出轮廓
    '''
    cv2.drawContours(img, contours, -1, (color), 2 )
    cv2.imshow(windowName,img)
    cv2.waitKey(0)
    
def getHorizon_vertical(img, hratio, vratio):
    '''
    @attention: 从图片中获取单个字的坐标
    @param charRatio: 字符宽度和图片高度的比例
    '''       
    #读取图片并获取基本信息
    H, W = img.shape[:2]
    #字符的大概宽度
    charLen = H * hratio
    #作为原图的逆图片
    imgSeg = 255 - img
    imgSeg[0:10, :] = 0
    imgSeg[H-10:H, :] = 0
    imgSeg[:, 0:10] = 0
    imgSeg[:, W-10:W] = 0
            
    horizon_sum =  np.sum(imgSeg, axis = 1)
    horizon_range = extract_peek_ranges_from_array(horizon_sum, charLen)
    #print('vertical_range', len(vertical_range))
    #print('horizon_range', len(horizon_range))
    ex = 2
    boxPoints = []
    texts = []
    hroi = img.copy()
    for posh, hr in enumerate(horizon_range):
        y1 = hr[0]
        y2 = hr[1]
        
        horizonROI = imgSeg[y1 - ex: y2 + ex, 0: W]
        #进行垂直投影
        vertical_sum = np.sum(horizonROI, axis = 0)
        vertical_range = extract_peek_ranges_from_array(vertical_sum, H * vratio)
        #个数不足，表明非标题
        if len(vertical_range) < 2:
            continue
        #重新获取y坐标
        ys = [y1, y2]
        #表明有多行，需要切割
        if (y2 - y1)/float(H) > 0.29:
            #print 'y2 , y1',y2 , y1
            for posv, vr in enumerate(vertical_range):
                x1 = vr[0]
                x2 = vr[1]
                VORI = imgSeg[y1 - ex: y2 + ex, x1: x2]
                #cv2.imshow('VORI',VORI)
                #cv2.waitKey(0)
                single_horizon_sum =  np.sum(VORI, axis = 1)
                single_horizon_range = extract_peek_ranges_from_array(single_horizon_sum, charLen)
                ys += [coor[0] + y1 - ex for coor in single_horizon_range]
                ys += [coor[1] + y1 - ex for coor in single_horizon_range]
        #去除重复的坐标并排序
        ys = list(set(ys))
        ys.sort()     
        #print 'ys',ys
        ystart = ys[0] # ys start
        posy = 1
        while posy < len(ys):
            yend = ys[posy] # ys end
            
            xmin = vertical_range[0][0]
            xmax = vertical_range[-1][1]
            boxPoints.append((xmin, ystart, xmax, yend))
                
            #在图上画上方框便于观察
            cv2.rectangle(hroi, (xmin, ystart), (xmax, yend), (0,0,255))
        
            posy += 1
            ystart = yend
            
    if len(boxPoints) == 0: 
        boxPoints = [(0, 0, W, H)]
            
    return boxPoints

def getConfig(filePath):
    '''
    @attention: 读取配置文件并获取配置
    '''
    doc = codecs.open(filePath, mode = 'r', encoding = 'utf-8')
    lines = doc.readlines()
    doc.close()
    
    configs = {}
    for line in lines:
        line = line.strip()
        
        if len(line) == 0 or line[0] == '#': 
            continue  #注释
        
        lineS = line.split('=')
        key = lineS[0].strip()
        value = lineS[-1].strip()
        configs[key] =value
    return configs

def getRegionFromTemplate(img, templatePath):
    '''
    @attention: 进行模板匹配，然后进行区域定位
    '''
    img2 = img.copy()
    #加载将要搜索的图像模板 
    tempImg = cv2.imread(templatePath, cv2.IMREAD_GRAYSCALE)
    #记录图像模板的尺寸 
    H, W = img.shape[:2]
    h, w = tempImg.shape[:2]
    if h > H or w > W:
        #print(u'img.rows >= templ.rows && img.cols >= templ.cols in function cv::matchTemplate')
        return None, None
    
    #pltShow([img2, tempImg], ['img2', 'tempImg'])
    
    #使用matchTemplate对原始灰度图像和图像模板进行匹配 
    #如果输入图像的大小是（WxH），模板的大小是（wxh），输出的结果的大小就是（W-w+1，H-h+1）。
    #当你得到这幅图之后，就可以使用函数cv2.minMaxLoc() 来找到其中的最小值和最大值的位置了。
    #第一个值为矩形左上角的点（位置），（w，h）为 moban 模板矩形的宽和高。这个矩形就是找到的模板区域了。
    res = cv2.matchTemplate(img2, tempImg, cv2.TM_CCOEFF) 
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, min(top_left[1] + h, H))
        
    return top_left, bottom_right

def templateAndOCR(img, templatePath):
    '''
    @attention: 进行模板匹配，然后进行区域定位
    '''
    H, W = img.shape[:2]
    #进行模板匹配
    top_left, bottom_right = getRegionFromTemplate(img, templatePath)
    if top_left is None:    return img
    #在ocrORI中的坐标
    ex = 5
    x1, y1 = top_left 
    x2, y2 = bottom_right
    x2 += x2 - x1
    
    x1 = max(x1 - ex, 0)
    x2 = min(x2 + ex, W)
    y1 = max(y1 - ex, 0)
    y2 = min(y2 + 1, H)
    ocrORI = img[y1:y2, x1:x2]
    
#     cv2.imshow('ocrORI',ocrORI)
#     cv2.waitKey(0)
    
    return ocrORI

def cvtImgName(dataPath, startValue = 0): 
    '''
    @attention: 将目标路径中所有的图片按照数字进行命名
    '''     
    fs = os.listdir(dataPath)
    fi = startValue 
    for f in fs:
        if f.split('.')[-1].lower() in ['jpg','png']:
            img = cv2.imread(dataPath + '/' + f)
            H,W = img.shape[:2]
            if H * W < 3000:
                img = cv2.resize(img, (W * 3, H * 3), interpolation = cv2.INTER_CUBIC)
            elif H * W > 100000:
                img = cv2.resize(img, (int(W * 0.3), int(H * 0.3)), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(dataPath + '/' + f, img)
            print(dataPath + '/' + f, dataPath + '/' + str(fi) +'.jpg')
            os.rename(dataPath + '/' + f, dataPath + '/' + str(fi) +'.jpg')
            fi += 1  
    return  fi

def addBoxFile(dataPath, originBoxFile, addBoxFile): 
    '''
    @attention: 将两个box文件合并
    '''     
    #获取原始box文件最后的数字代码
    originDoc = codecs.open(dataPath + '/' + originBoxFile, mode = 'r', encoding = 'utf-8')
    originlines = originDoc.readlines()
    originlines = [x for x in originlines if x.strip() != '']
    lastNum = int(originlines[-1].split(' ')[-1])
    #print ''.join(originlines)
    print('lastNum',lastNum)
    originDoc.close()
    #对添加文件进行处理，获取其内容并把页码数添加lastNum+1
    addDoc = codecs.open(dataPath + '/' + addBoxFile, mode = 'r', encoding = 'utf-8')
    addlines = addDoc.readlines()
    #addlines = [x for x in originlines if x.strip() != '']
    for line in addlines:
        if line.strip() == '': continue
        chars = line.split(' ')
        lastChar = str(int(chars[-1]) + lastNum + 1) + u'\r\n'
        chars[-1] = lastChar
        newLine = u' '.join(chars)
        originlines.append(newLine) 
    #print ''.join(originlines)
    addDoc.close()
    #将最后的列表写人新的文件
    writeDoc = codecs.open(dataPath + '/new.font.exp0.box', mode = 'w', encoding = 'utf-8')
    writeDoc.writelines(originlines)
    writeDoc.close()
    print(dataPath + '/new.font.exp0.box is done')  



def findSpecifyPic(dataPath):
    '''
    @attention: 将图片复制到指定的目录中
    '''       
    names = u'''100965913-100965913-00000102467117674-8000.jpg 1 TY014
102573994-102573994-9163890018534025-23.75.jpg 1 TY414
'''
    os.chdir(dataPath)
    desPath = dataPath + '/modify/'
    if os.path.exists(desPath):
        shutil.rmtree(desPath)
    os.mkdir(desPath)
    
    names = names.split('\n')
    names = [x.split(' ')[0] for x in names]
    print(names)
    for name in names:
        if os.path.exists(dataPath + '/' + name):
            shutil.copy(dataPath + '/' + name, desPath + name)
            
def testTrainData(dataPath, langName, types):
    '''
    @attention: 测试字典
    '''     
    fs = os.listdir(dataPath)
    
    for f in fs:
        if f.split('.')[-1].lower() in ['jpg','png']:
            img = cv2.imread(dataPath + '/' + f)
            H,W = img.shape[:2]
            if H * W < 20000:
                img = cv2.resize(img, (W * 3, H * 3), interpolation = cv2.INTER_CUBIC)
            text = handleText(ocrIdCard(img, langName))
            type = justZyTitle(text, types)
            #flag = justTest(text)
            #print f.split('.')[0],"\t'", text, "\t", type
            
def justInExitTypes(tit):
    '''
    @attention: 测试字是否在词库中
    '''  
    types = u'''存款凭条取银行卡记账证现金管理收费联来缴单特殊业务申请书挂失电子转客户资料登变更表回利息及代扣税清身份信网
    查询交易支票汇山东省通公司专用发内往系统出复到柜员日终平帐报告重要空白销号贷方中国农购受综合应西穗换医疗保险知小额付待处总短服再
    生源供货同签约入库卷别明细执两码确认笔核对结果天津分进储蓄所兑差消惠借个人买外审批民打印算调开放式基项厦门市验预留鉴种传涉科目协
    议跨错商承期领冲正住房积移币化补贴定止位道路技术监控违法为自助境机构大动集团有限拨非济南品宅维修第三企话手上密器使抵押漫游事一据
    债植城风能力评估问护口融计整注册须划安简程序罚决未提前归还改红活本般社会工经残疾障名退详情投诉建立络齐鲁招实物黄海浦展按揭财产授
    权委托候政封面居售假说箱租券过渡类型拔点营量余运况双丰者由粘救直接阅录贸素卖成片副食价格节撤示函讼体照美元钞即普遇影像妥善部零低
    质策性职标志章数字关组织并落地赎折浙江称先准'''
    for ti in tit:
        if ti in types:
            print('in',ti )
        else:
            print('not in ',ti)

def getIMGHeightWidth(dataPath, isHbigger = True):
    '''
    @attention: 获取文件夹中图片的长宽
    ''' 
    #基本参数设置
    from PIL import Image
    print('-'*30 + dataPath)
    listPics = os.listdir(dataPath)#图片列表
    Hs, Ws, rs = [], [], []
    #创建保存标题的文件夹
    for pos, pic in enumerate(listPics):
        #如果非图片
        if os.path.splitext(pic)[-1].lower() not in ['.jpg', '.png']: continue
        imgPath = dataPath + '/' + pic
        #img = cv2.imdecode(np.fromfile(imgPath, dtype = np.uint8), -1)
        #H, W   = img.shape[:2]
        img = Image.open(imgPath)
        H , W = img.height, img.width
        if isHbigger and H < W: 
            print(pic + ' H, W\t', H , '\t' , W , '\t ratio:\t', float(H) / W)
            continue
        elif not isHbigger and H > W:
            print(pic + ' H, W\t', H , '\t' , W , '\t ratio:\t', float(H) / W)
            continue
        Hs.append(H)
        Ws.append(W)
        rs.append(float(H) / W)
        #print pic + ' H, W\t', H , '\t' , W , '\t ratio:\t', float(H) / W
        
        if pos == 5000: break
    try:   
        print(pos)
        print('min(H), max(H):\t',min(Hs), '\t', max(Hs), '\t',  np.average(Hs))
        print('min(W), max(W):\t',min(Ws), '\t', max(Ws), '\t',  np.average(Ws))
        print('min(r), max(r):\t',min(rs), '\t', max(rs))
        #print 'len(Hs)\t',len(Hs), 'pos\t', pos, 'ratio\t', len(Hs) / pos
    except ValueError:
        print('min() arg is an empty sequence')
    
def inRange(data, ra):
    '''
    @attention: 判断是否在ra这个数据范围中
    '''     
    if data <= ra[1] and data >= ra[0]:
        return True
    return False

def createOutDir(dataPath, dirs):
    '''
    @attention: 创建输出的文件夹
    @param dataPath: 文件夹
    '''
    #保存临时图片路径
    savePathDict = {}
    while 1:
        try:
            for dir in dirs:
                savePath = dataPath + dir
                savePathDict[dir] = savePath
                if os.path.exists(savePath): shutil.rmtree(savePath)
                os.mkdir(savePath)
            break
        except WindowsError:
            #print 'window error, try again'
            pass
    #print('-----------------createOutDir is done!')
        
    return savePathDict

def findLines(img, minLength = 300):
    '''
    @attention: 在图片中找到水平直线
    '''  
    if len(img.shape) > 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img = cv2.GaussianBlur(img,(3,3),0)
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    #HoughLines：第二和第三个值分别代表 ρ 和 θ 的精确度。第四个参数是阈值，只有累加其中的值高于阈值时才被认为是一条直线，也可以把它看成能检测到的直线的最短长度（以像素点为单位）
    lines = cv2.HoughLines(edges,1,np.pi/180,minLength) #这里对最后一个参数使用了经验型的值
    result = img.copy()
    ys = []
    #如果有直线则处理
    if isinstance(lines, np.ndarray):
        for line in lines[0]:
            rho = line[0] #第一个元素是距离rho
            theta= line[1] #第二个元素是角度theta
            if  (theta > (15.*np.pi/32.0 )) and (theta < (17.*np.pi/32.0)): #水平直线, 0.5pi
                # 该直线与第一列的交点
                #print theta
                pt1 = (0,int(rho/np.sin(theta)))
                #该直线与最后一列的交点
                pt2 = (result.shape[1], int((rho-result.shape[1]*np.cos(theta))/np.sin(theta)))
                #print u'水平',pt1, pt2, theta, theta * 180
                
                avgy = 0.5*(pt1[1]+ pt2[1])
                ys.append([pt1, pt2, theta, avgy])
        #从上至下排序
        ys.sort(key=lambda ys:ys[3])
        
    #删除相邻的直线
    pos = 0
    while pos< len(ys):
        pt1_, pt2_, theta_, avgy_ = ys[pos]
        #删除太靠上的直线
        if pt1_[1] < 40 or pt2_[1] < 40:
            ys.pop(pos)
            continue
        #删除相邻的直线
        if pos+1 < len(ys):
            pt1, pt2, theta, avgy = ys[pos + 1]
            if abs(avgy - avgy_) < 20:
                ys.pop(pos)
                pos -= 1
        pos += 1
    
    #绘制一条直线
    # for pt1, pt2, theta, avgy in ys:    cv2.line(result, pt1, pt2, (0), 2)
    #pltShow([img,img, edges, result ],['img1', 'img', 'edges','result' ])     
    return ys


def horizonImg(img, minLength = 180):
    '''
    @attention: 在图片中根据顶部的直线的角度水平校正
    ''' 
    ys =  findLines(img, minLength)
    #如果没有直线则返回 原图
    if len(ys) == 0: return img   
    #获取弧度，这里需要转换角度
    angle =  ys[0][2]
    angle = 90 - angle * 180 / np.pi  
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    #print 'angle', angle
    # 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子,可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    M=cv2.getRotationMatrix2D((0,0), angle, 1.0)
    # 第三个参数是输出图像的尺寸中心
    rows, cols = img.shape[:2]
    #print 'rows, cols, angle',rows, cols, angle
    dst = cv2.warpAffine(img, M,(cols,rows), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)    
    #pltShow([img,dst],['img','dst'])
    
    return dst

def inPaintLine(img, minLength = 180):
    '''
    @attention: 在图片中找到直线并消除
    ''' 
    temp = img.copy()   
    ys =  findLines(temp, minLength)
    if len(ys) == 0: return temp
    
    for pt1, pt2, theta, avgy in ys:
        y0, angle = max(pt1[1], pt2[1]), theta
        #消除下划线
        temp[y0 - 3 :y0 + 4, :] = 255
    #pltShow([img,temp],['img','temp'])
        
    return temp

def oriByTopLines(img, minLength = 300,isCut=True):
    '''
    @attention: 在图片中根据顶部的直线划取顶部标题图片，并通过角度水平校正
    '''    
    ys =  findLines(img, minLength)
    if len(ys) == 0: return img
    
    y0, angle = 0, 0
    for pt1, pt2, theta, avgy in ys:
        y0, angle = max(pt1[1], pt2[1]), theta
        if y0 < 130:    continue
        break
    #要往上提
    y0 -= 4
    #截取ori并进行角度校正
    if isCut:
        ori = img[0:y0, :]
    else:
        ori = img
    H, W = ori.shape[:2]
    if H < 110: ori = img[0:110, :]  
    
    return ori


def getDiffColorImg(imgori,isCut=True):
    H, W = imgori.shape[:2]
    
    imgs = []
    ratioDist = {}
    if H >= 85:
        #img = imgori[:int(H * 0.97), :]
        img = imgori.copy()
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, grayImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        #bgr分割
        bimg, gimg, rimg = cv2.split(img)
        retval, bimg = cv2.threshold(bimg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        # retval, gimg = cv2.threshold(gimg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        retval, rimg = cv2.threshold(rimg, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        #灰色部分
        noGrayImg = cv2.add(grayImg, cv2.inRange(hsvImg, np.array([45,0,46]), np.array([180,43,220])))
        #求和
        sumg_grayImg = np.sum(grayImg)
        
        #求取比例
        ratioDist[0] = grayImg
        ratioDist[(np.sum(bimg) * 1.0 / sumg_grayImg - 1) * 1000] = bimg
        ratioDist[(np.sum(rimg) * 1.0 / sumg_grayImg - 1) * 1000] = rimg
        ratioDist[(np.sum(noGrayImg) * 1.0 / sumg_grayImg - 1) * 1000] = noGrayImg
        #pltShow([grayImg, bimg, rimg, noGrayImg],['grayImg', 'bimg','rimg', 'noGrayImg'])
   
        #去除小于0 的比例
        repDist = {} #优先的选择
        leftDist = {} #剩余的选择
        for ratio, imgC in list(ratioDist.items()):
            if ratio < 0 or ratio > 80:     ratioDist.pop(ratio)
            elif inRange(ratio, (0, 5)):    repDist[ratio] = imgC
            elif ratio > 12: leftDist[ratio] = imgC
        #对repDist进行排序, 选择变化最大的那个
        repkeys =list(repDist.keys())
        repkeys.sort()
        repImg = repDist[repkeys[-1]]
        #对 leftDist 进行排序
        leftkeys = list(leftDist.keys())
        leftkeys.sort(reverse = True)
        #如果图片过大，还需要进行划分
        if isCut and H >= 105: 
            imgs += [repImg[:int(0.6*H),:], repImg[int(0.4*H):,:]]
            if  len(leftkeys) > 0:
                for k in leftkeys: 
                    imgs += [leftDist
                             [k][:int(0.6*H),:], leftDist[k][int(0.4*H):,:]] 
            
                imgs.append(leftDist[leftkeys[0]])
            else:
                imgs.append(repImg)
        else:   
            imgs.append(repImg)
            imgs += [leftDist[k] for k in leftkeys]       
        
        #print repDist.keys(), leftDist.keys(), len(imgs)
    
    else: imgs = [imgori]
            
    return imgs        
             
def getSpaceCombination(key, ps):
    '''
    @attention: 对类型名称key， 采用缺额位置进行插入，生成正则表达式
    '''
    patterns = []
    for i,pl  in enumerate(ps):
        keys = [x for x in key]
        for j, pc in enumerate(pl):
            keys[pc] = u'P'
        remain = 'P'.join(keys)
        pattern = remain.replace('PPPP', u'.{0,4}')
        pattern = pattern.replace('PPP', u'.{0,3}')
        pattern = pattern.replace('PP', u'.{0,2}')
        pattern = pattern.replace('P', u'.{0,1}')
        #print i, remain, '\t',pattern
        patterns.append(pattern)
    
    return patterns
        
def getPosition(num):
    '''
    @attention: 对数字长度进行组合
    '''
    ps = []
    if num<=6 and num >= 5:
        #只能有一个错误
        for i in range(num):    ps.append([i])
    elif num<=9 and num >= 7:
        #只能有两个错误
        for i in range(num):   
            for j in range(i+1, num): 
                ps.append([i, j])
    elif num >= 10:
        #只能有4个错误
        for i in range(num):   
            for j in range(i+1, num): 
                for m in range(j+1, num):
                    for n in range(m+1, num):
                        ps.append([i, j, m, n])
    
    return ps

def segStr(cstr):
    '''
    @attention: 长的字符串的分割
    '''
    lines = cstr.split('\n')
    for line in lines:
        line = line.strip().split(' ')
        for pos, li in enumerate(line):
            if pos % 51 == 0:
                line.insert(pos, 'P')
        line = ''.join(line)
        line = line.replace('P', '\\\n\t')
        #print line
        
def getTypes_(filePath):
    '''
    @attention: 读取配置信息，获取类型对应表
    @param filePath: 配置文件路径 
    @return: 返回类型对应dist
    '''  
    #读取文件获取类型对应表
    #filePath = os.path.dirname(__file__) + '/billType.cfg'
    doc = codecs.open(filePath, mode = 'r', encoding = 'utf-8')
    typeStr =  ''.join(doc.readlines())
    doc.close()
    
    temp_types = []
    typeStr = typeStr.split('\n')
    for ai in typeStr:
        if len(ai.strip()) == 0:    continue
        ls = ai.split('\t')
        key = ls[0].strip()
        value = ls[1].strip()
        #print key, '\t', value
        temp_types.append([len(key), key, value])
    temp_types.sort(key=lambda temp_types:temp_types[0], reverse=True)
    
    types = []
    for pos, item in enumerate(temp_types):
        lenKey, key, value = item[0], item[1], item[2]
        #获取key对应的正则表达式
        if len(key) >= 5 and len(key) <= 11:
            patterns = getSpaceCombination(key, getPosition(len(key)))
            for pattern in patterns:
                types.append([lenKey, pattern, value])
        else:
            types.append([lenKey, key, value])
    
    return types

def justZyTitle_(res, types):
    '''
    @attention: 判断字符串res属于那种标题
    @param types: 判断类型列表
    @return: 返回对应标题
    ''' 
    res = handleZYtitle(res)
    #print res
    #采用精确查询
    for item in types:
        key, value = item[1], item[2]
        if key in res:      return value
    #如果精确查询没有找到，采用模糊查询
    for item in types:
        lenKey, key, value = item[0], item[1], item[2]
        num = 0.0
        #如果字数大于11个,选取包含字数概率大于0.7且选最大的概率
        if len(key) > 11:
            tstr = res
            akey = u''
            for pk, ki in enumerate(key):
                if ki in tstr:   
                    num += 1
                    ind = tstr.index(ki)
                    tstr = tstr[ind:]
                    akey+=ki
                akey += u'.{0,3}'
            ratio = num / len(key)
            if ratio > 0.7 and ratio > maxRatio and len(re.findall(akey, res)) > 0:    
                maxRatio, proKey, proValue = ratio, key, value
        #如果字数大于5个的情况，采用正则表达式来判断， 正则表达式为key， 其中5-6个字只允许错1个字，7-9允许错2个字， 10-12允许4个字
        elif len(key) >= 5:
            #如果字数为13字以上的， 且概率足够大，则返回类型值
            if maxRatio > 0.7:  return proValue
            if len(re.findall(key, res)) > 0:     
                if value == u'境内-外汇款申请书' and (u'客户' in res or u'人民' in res):   value = u'个人客户境内汇款申请书'
                if value == u'特种转账传票' and u'进账' in res: value = u'银行进账单'
                if u'财政' in key and u'财政资' in res:      value = u'财政资金支付凭证'
                if key == u'财政资金支付凭证':
                    if u'政性' in res or u'性资' in res:      value = u'财政性资金支付凭证'
                    if u'授权支付' in res  :      value = u'财政授权支付凭证'                
                return value
        #如果字数为4个的情况    
        elif len(key) == 4:
            #对可能存在的情况进行列举
            if len(re.findall(u'(政直接支)|(直接支付)|(财政直.{0,3}接支)', res)) > 0:     return u'财政直接支付凭证'
            if len(re.findall(u'(凭.{0,4}收账通知)', res)) > 0:     return u'财政资金支付凭证'
            if len(re.findall(u'(授.{0,1}权.{0,4}凭证)|(授.{0,1}权.{0,4}支付凭)|(财政授.{0,1}权)|(政授.{0,1}权支)', res)) > 0:     return u'财政授权支付凭证'
            if len(re.findall(u'(999.{0,6}欠.{0,1}凭)|(999.{0,6}存.{0,1}欠)|(^999.{0,1}临)|(^999.{0,1}合)|(^999.{0,3}存)|(^999.{0,6}欠)', \
                              re.sub(u'[1-8]', u'', res))) > 0:     return u'9991临时存欠凭证'
            if len(re.findall(u'(^结算业务)|(算业务申)|(^结算.{0,1}务申)', re.sub(u'[0-9]', u'', res))) > 0:     return u'结算业务申请书'
            if len(re.findall(u'(个.{0,1}开户及综)|(个人开户及)|(个人开.{0,4}综合服.{0,2}申)',  res)) > 0:     return u'个人开户及综合服务申请表'
            if len(re.findall(u'(业务种类行.{0,4}款)|(业务种类行内)|(业务种类行.{0,4}汇)',  res)) > 0:     return u'结算业务申请书'
            #对5-6的key中现金缴款单和特种转账传票进行处理
            if len(re.findall(u'(现.{0,3}缴款)|(^.{0,7}现.{0,3}缴.{0,1}单)|(个正人.{0,4}缴款)', res)) > 0:      return u'现金缴款单'
            if len(re.findall(u'(款通知.{0,5}转账)|(种转账传)|(转账传票)', res)) > 0:      return u'特种转账传票'
            if len(re.findall(u'(^外汇兑换)|(^外汇单换)|(^外汇兑挂支)', res)) > 0:      return u'外汇兑换水单'
            if len(re.findall(u'(银行单位存)|(^单位存.{0,1}算.{0,3}人凭条)|(单位存.{0,3}凭.{0,3}期)', res)) > 0:      return u'单位存款凭条'
            
            #用正则表达式进行判断4个字情况
            pattern = u'.{0,1}'.join(key)
            #print pattern, u''.join(re.findall(pattern, res))
            if len(re.findall(pattern, res)) > 0:      return value
            continue

        #如果类型为进账单
        elif key == u'进账单':
            #对4字的key中现金支票和转账支票进行处理
            xjzpPa = u'(^.{0,10}现金.{0,2}支.{0,2}票)|(银行.{0,3}金支)|(^.{0,10}明正.{0,1}支票)'
            xjzpPa += u'|(^.{0,10}现.{1,2}支票)|(^.{0,10}现金支.{1,2})|(.{1,2}金支票)|(^.{0,10}现金.{1,2}票)'
            zzhpPa = u'(^.{0,10}转.{1,2}支票)|(^.{0,10}转账支.{1,2})|(^.{0,10}账支票)|(^.{0,10}转账.{0,2}票)'
            if len(re.findall(xjzpPa, res)) > 0 and len(re.findall(u'方凭证', res)) == 0 :
                if len(res) > 15 or len(re.findall(u'现金缴票', res)) < 1:   return u'现金支票'
            if len(re.findall(zzhpPa, res)) > 0 and len(re.findall(u'方凭证', res)) == 0 :
                if len(res) > 15 or len(re.findall(u'^特', res)) < 1: return u'转账支票'
            #用正则表达式进行判断
            if len(re.findall(u'(进.{0,1}账单)|(进账.{0,1}单)', res)) > 0:   return value
            continue
    #疑似存折的情况
    if len(re.findall(u'\d{18}', res)) > 0 and res.count('1') < 8:     return u'no'  
    
    return u'None'

def getTypeRatio(filePath):
    '''
    @attention: 读取配置信息，获取类型对应的高度/宽度比例
    @param filePath: 配置文件路径 
    @return: 返回类型对应dist
    '''  
    #读取文件获取类型对应表
    #filePath = os.path.dirname(__file__) + '/billType.cfg'
    doc = codecs.open(filePath, mode = 'r', encoding = 'utf-8')
    typeStr =  ''.join(doc.readlines())
    doc.close()
    
    typeRatios = {}
    typeStr = typeStr.split('\n')
    for ai in typeStr:
        if len(ai.strip()) == 0:    continue
        ls = ai.split('\t')
        key = ls[0].strip()
        minR = float(ls[1].strip())
        maxR = float(ls[2].strip())
        #print key, '\t', minR, maxR
        typeRatios[key] = [minR, maxR]
    
    return typeRatios

def justBlankOrBlack(img, minR = 0.02):
    '''
    @attention: 通过计算图形的和判断是否是空白图还是主体为黑色污染的图
    @param img: 原图像， 应为二值化图,白底黑字
    @param minR: 黑色部分占的最小比率
    '''    
    H, W = img.shape[:2]
    #求取黑色区域比例
    ratio = 1 - np.sum(img) / (255.0 * H * W)  #计算黑色区域所占的比例
    #print 'ratio',ratio
    #比例过大或者过小
    if ratio < minR or ratio > 0.65:
        return True
    return False


    