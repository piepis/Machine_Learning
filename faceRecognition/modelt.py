#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:modeltest.py
@time:2018-04-109:28
@desc:
'''


from read_data import read_name_list,read_file
from train_model import Model
import cv2
from keras import backend as K
K.set_image_dim_ordering('tf')

def onePicture(path):
    model= Model()
    model.load()
    img = cv2.imread(path)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    picType,prob = model.predict(img)
    if picType != -1:
        name_list = read_name_list('webface')
        #print(name_list)
        print( name_list[picType],prob)
    else:
        print (" Don't know this person")

#读取文件夹下子文件夹中所有图片进行识别
def onBatch(path):
    model= Model()
    model.load()
    index = 0
    img_list, label_lsit, counter = read_file(path)
    for img in img_list:
        picType,prob = model.predict(img)
        if picType != -1:
            index += 1
            name_list = read_name_list('F:\myProject\pictures\dataset')
            print(name_list)
            print (name_list[picType])
        else:
            print (" Don't know this person")
    return index

if __name__ == '__main__':
    onePicture(r'009.jpg')
    #test_onBatch('D:\myProject\pictures\pic4')



