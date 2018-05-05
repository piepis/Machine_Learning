#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:DLModel.py
@time:2018-03-1413:47
@desc:车牌识别系统中所有用到的model
'''

from keras.models import Model,Input,Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten,BatchNormalization, Conv2D,MaxPool2D
from keras import backend as K
K.set_image_dim_ordering('tf')
from .CharsCode import e2e_chars as chars
#--------------------------------------------
def Getmodel_tensorflow0(nb_classes):
    img_rows, img_cols = 9, 34
    nb_pool = 2
    model = Sequential()
    model.add(Conv2D(16, (5, 5),input_shape=(img_rows, img_cols,3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
plate_type_model = Getmodel_tensorflow0(5)    #td.SimplePredict(plate) 中用到的 model
plate_type_model.load_weights("./model/plate_type.h5")  #初始化一个完全相同的模型
#----------------------------------------------
def getModel():
    input = Input(shape=[16, 66, 3])  # change this shape to [None,None,3] to enable arbitraty shape input
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = Activation("relu", name='relu1')(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = Activation("relu", name='relu2')(x)
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = Activation("relu", name='relu3')(x)
    x = Flatten()(x)
    output = Dense(2,name = "dense")(x)
    output = Activation("relu", name='relu4')(output)
    model = Model([input], [output])
    return model
model12_model = getModel()
model12_model.load_weights("./model/model12.h5") #模型加载初始化权重
#  ----------------------------
def construct_model(model_path):
    input_tensor = Input((None, 40, 3))
    x = input_tensor
    base_conv = 32
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
ocr_plate_all_w_rnn_2_model = construct_model("./model/ocr_plate_all_w_rnn_2.h5",)
#---------------------------
def Getmodel_tensorflow1(nb_classes):
    img_rows, img_cols = 23, 23
    nb_filters = 16
    nb_pool = 2
    nb_conv = 3
    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model
char_judgement_model  = Getmodel_tensorflow1(3)
char_judgement_model.load_weights("./model/char_judgement.h5")
#----------------------------------
def Getmodel_tensorflow2(nb_classes):
    # nb_classes = len(charset)
    img_rows, img_cols = 23, 23
    # number of convolutional filters to use
    # nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    # nb_conv = 3
    # x = np.load('x.npy')
    # y = np_utils.to_categorical(range(3062)*45*5*2, nb_classes)
    # weight = ((type_class - np.arange(type_class)) / type_class + 1) ** 3
    # weight = dict(zip(range(3063), weight / weight.mean()))  # 调整权重，高频字优先
    model = Sequential()
    model.add(Conv2D(32, (5, 5),input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
def Getmodel_ch(nb_classes):
    # nb_classes = len(charset)
    img_rows, img_cols = 23, 23
    # number of convolutional filters to use
    # nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    # nb_conv = 3
    model = Sequential()
    model.add(Conv2D(32, (5, 5),input_shape=(img_rows, img_cols,1)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(756))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
char_rec_model  = Getmodel_tensorflow2(65)
char_chi_sim_model_ch = Getmodel_ch(31)
char_chi_sim_model_ch.load_weights("./model/char_chi_sim.h5")
char_rec_model.load_weights("./model/char_rec.h5")