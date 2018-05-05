#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:FacePlateRecognitionMain.py
@time:2018-03-1315:22
@desc: 对车牌进行预测
'''
import numpy as np
import cv2
from .DLModel import ocr_plate_all_w_rnn_2_model as pred_model
from .CharsCode import e2e_chars as chars
def fastdecode(y_pred):
    results = ""
    confidence = 0.0
    table_pred = y_pred.reshape(-1, len(chars)+1) #1代表沿着行的  沿轴axis最大值的索引。
    res = table_pred.argmax(axis=1)
    for i,one in enumerate(res):
        if one<len(chars) and (i==0 or (one!=res[i-1])):
            results+= chars[one]
            confidence+=table_pred[i][one]
    confidence/= len(results)
    return results,confidence
def recognizeOne(x_tempx): #车牌预测函数
    x_temp = cv2.resize(x_tempx,( 160,40)) #对图片进行扩展缩放，指定图片大小
    x_temp = x_temp.transpose(1, 0, 2) #对图像进行转置 转换 90 度
    y_pred = pred_model.predict(np.array([x_temp])) #加载model 训练集 进行预测
    y_pred = y_pred[:,2:,:]
    return fastdecode(y_pred)
