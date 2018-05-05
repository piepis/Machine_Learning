#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:FacePlateRecognitionMain.py
@time:2018-03-1315:22
@desc:对矫正后的图像进行预测
'''
import numpy as np
import cv2
from .DLModel import model12_model as model
def finemappingVertical(image):
    resized = cv2.resize(image,(66,16))
    resized = resized.astype(np.float)/255
    res= model.predict(np.array([resized]))[0]
    res  =res*image.shape[1]
    res = res.astype(np.int)
    H,T = res
    H-=3
    if H<0:
        H=0
    T+=2
    if T>= image.shape[1]-1:
        T= image.shape[1]-1
    image = image[0:35,H:T+2]
    image = cv2.resize(image, (int(136), int(36)))
    return image