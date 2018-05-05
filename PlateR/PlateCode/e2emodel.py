#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:FacePlateRecognitionMain.py
@time:2018-03-1315:22
@desc:保存的训练模型
'''
from keras.models import Model,Input
from keras.layers import Activation,BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
def construct_model(model_path):
    input_tensor = Input((None, 40, 3))
    x = input_tensor
    base_conv = 32
    from .e2e import chars
    for i in range(3):
        x = Conv2D(base_conv * (2 ** (i)), (3, 3),padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (5, 5))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, (1, 1))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(len(chars)+1, (1, 1))(x)
    x = Activation('softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)
    base_model.load_weights(model_path)  #载入训练模型
    return base_model
