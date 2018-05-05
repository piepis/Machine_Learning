#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:FacePlateRecognitionMain.py
@time:2018-03-1315:22
@desc:  车牌号识别系统的主函数
'''
from . import detect
from . import  finemapping  as  fm
from . import segmentation
import cv2
from . import typeDistinguish as td
from . import e2e
from . import cache
from . import finemapping_vertical as fv
def SimpleRecognizePlate(image):
    images = detect.detectPlateRough(image,image.shape[0],top_bottom_padding_rate=0.1)#得到可能是车牌号的图片的，大范围 ，原图上的坐标点位置 ，小返回，
    dictplate=[]
    if len(images)>0:
        for j,plate1 in enumerate(images): #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            plate, rect, origin_plate  =plate1   # 大范围 ，原图上的坐标点位置 ，小返回，
            cv2.imwrite('test1.jpg', plate)
            plate  =cv2.resize(plate,(136,36*2))
            cv2.imwrite('test2.jpg', plate)
            ptype = td.SimplePredict(plate)
            if ptype>0 and ptype<5:
                plate = cv2.bitwise_not(plate)
            image_rgb = fm.findContoursAndDrawBoundingBox(plate) #矫正图像
            image_rgb = fv.finemappingVertical(image_rgb) # 矫正后的图像进行预测，然后放大输出 里面进行了一次keras 的预测
            image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY) #将色域去掉

            ETEStr, ETEResult = e2e.recognizeOne(image_rgb) #端到端检验 返回 预测结果和车牌号

            blocks, CApartStr, CApartresult = segmentation.slidingWindowsEval(image_gray)  # 分割识别

            if ETEStr == CApartStr or (ETEResult + CApartresult/7)/2 >0.3:
                if ETEStr == CApartStr:
                    cache.verticalMappingToFolder(image_rgb)  # 保存图片
                    print("车牌:",ETEStr,"置信度:",(ETEResult + CApartresult/7)/2)
                    dictplate.append((j,ETEStr))
                elif ETEResult> CApartresult/7:
                    print("车牌可能是:{0} 或者是 {1}".format(ETEStr,CApartStr))
                    dictplate.append((j,ETEStr))
                else:
                    dictplate.append((j,CApartStr))
            else:
                print("图片模糊或者没有车牌")
                dictplate.append((j,None))

    else:
        print("该图片中没有检测到车牌，请重新上传一张清晰的车牌")
        dictplate.append((None, None))
    return dictplate




if __name__ == '__main__':
    SimpleRecognizePlate(image = "../11.jpg")





