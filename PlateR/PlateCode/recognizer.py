#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:recognizer.py
@time:2018-03-1315:22
@desc:对矫正后的图像进行预测
'''
import cv2
import numpy as np
from .DLModel import (char_chi_sim_model_ch as model_ch,char_rec_model as model)
from .CharsCode import Rzer_chars as chars
def SimplePredict(image,pos):
    image = cv2.resize(image, (23, 23))
    image = cv2.equalizeHist(image)
    image = image.astype(np.float) / 255
    image -= image.mean()
    image = np.expand_dims(image, 3)
    if pos!=0:
        res = np.array(model.predict(np.array([image]))[0])
    else:
        res = np.array(model_ch.predict(np.array([image]))[0])
    zero_add = 0
    if pos==0:
        res = res[:31]
    elif pos==1:
        res = res[31+10:65]
        zero_add = 31+10
    else:
        res = res[31:]
        zero_add = 31
    max_id = res.argmax()
    return res.max(),chars[max_id+zero_add],max_id+zero_add

