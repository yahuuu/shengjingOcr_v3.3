- 环境搭建
#默认已经完成tesseraact的环境的搭建。具体搭建方法请参考之前发送的文件。
python3.6.2的依赖版本请按照requirements.txt
"""
pip install -r requirements.txt
"""

- 运行注意
#如果出现：!strcmp(locale, "C"):Error:Assert failed:in file baseapi.cpp, line 209  段错误(吐核)
#在使用python输入以下命令行
"""
export LC_ALL=C
"""
运行服务
"""
bash start_webService_2.sh
"""
关闭服务
"""
bash close_webService.sh
"""

- 运行可能出现问题的解决方法
# tess环境搭建后如果单独运行billTypeWebService_v2_sub.py提示缺少底层某个.so链接库，请尝试替换安装路径中sjocr_v3.0/sjocr/sjocr/ocr_models/Tesseract_API/linux中的libtesseract* 文件。


- 版本更新
"""
# 3.2 更新说明：
1. 流程最后增加模板匹配
2. 新增加的类别为：065， 066， 532， 520 

# 3.1 更新说明：
1. billTypeWebService_v2_sub.py 添加 通用凭证 201映射
2. billTypeInfo.cfg	添加 通用凭证 类型
3. billTitleOCR.py 202行 修改轮廓 过滤条件 由0.6 换成 0.7 通用凭证-00000019.JPG

# 3.0 更新说明：
1. cnn 集成到版面识别前
2. 输出类型转化为代号
3. 语言升级到python3.6.2

# 2.1更新说明：
1. 增加了log
2. 修改了二值化方法
3. 结算业务申请书，有符号无符号判断
"""
