#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:FacePlateRecognitionMain.py
@time:2018-03-1315:22
@desc:
'''
import cv2
import numpy as np
from .DLModel import plate_type_model as model
def SimplePredict(image):
    image = cv2.resize(image, (34, 9))
    cv2.imwrite('test3.jpg', image)
    image = image.astype(np.float) / 255
    cv2.imwrite('test4.jpg', image)
    res = np.array(model.predict(np.array([image]))[0])
    return res.argmax()


