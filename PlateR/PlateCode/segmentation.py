#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:FacePlateRecognitionMain.py
@time:2018-03-1315:22
@desc: 分割识别
'''
import cv2
import numpy as np
import scipy.ndimage.filters as f
import scipy.signal as l
from .DLModel import char_judgement_model as model2
from . import recognizer as cRP
def get_median(data):
   data = sorted(data)
   size = len(data)
   # print size
   if size % 2 == 0: # 判断列表长度为偶数
    median = (data[size//2]+data[size//2-1])//2
    data[0] = median
   if size % 2 == 1: # 判断列表长度为奇数
    median = data[(size-1)//2]
    data[0] = median
   return data[0]
def searchOptimalCuttingPoint(rgb,res_map,start,width_boundingbox,interval_range):
    length = res_map.shape[0]
    refine_s = -2
    if width_boundingbox>20:
        refine_s = -9
    score_list = []
    interval_big = int(width_boundingbox * 0.3)  #
    p = 0
    for zero_add in range(start,start+50,3):
        # for interval_small in xrange(-0,width_boundingbox/2):
        for i in range(-8,int(width_boundingbox/1)-8):
            for refine in range(refine_s, int(width_boundingbox/2+3)):
                p1 = zero_add# this point is province
                p2 = p1 + width_boundingbox +refine #
                p3 = p2 + width_boundingbox + interval_big+i+1
                p4 = p3 + width_boundingbox +refine
                p5 = p4 + width_boundingbox +refine
                p6 = p5 + width_boundingbox +refine
                p7 = p6 + width_boundingbox +refine
                if p7>=length:
                    continue
                score = res_map[p1][2]*3 -(res_map[p3][1]+res_map[p4][1]+res_map[p5][1]+res_map[p6][1]+res_map[p7][1])+7
                # print score
                score_list.append([score,[p1,p2,p3,p4,p5,p6,p7]])
                p+=1
    score_list = sorted(score_list , key=lambda x:x[0])
    return score_list[-1]
def niBlackThreshold( src,  blockSize, k  ):
    mean = cv2.boxFilter(src,cv2.CV_32F,(blockSize, blockSize),borderType=cv2.BORDER_REPLICATE)
    sqmean = cv2.sqrBoxFilter(src, cv2.CV_32F, (blockSize, blockSize), borderType = cv2.BORDER_REPLICATE)
    variance = sqmean - (mean*mean)
    stddev  = np.sqrt(variance)
    thresh = mean + stddev * float(-k)
    thresh = thresh.astype(src.dtype)
    k = (src>thresh)*255
    k = k.astype(np.uint8)
    return k
def refineCrop(sections,width=16):
    new_sections = []
    for section in sections:
        sec_center = np.array([section.shape[1]/2,section.shape[0]/2])
        binary_niblack = niBlackThreshold(section,17,-0.255)
        imagex, contours, hierarchy  = cv2.findContours(binary_niblack,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        boxs = []
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            ratio = w/float(h)
            if ratio<1 and h>36*0.4 and y<16 :
                box = [x,y,w,h]
                boxs.append([box,np.array([x+w/2,y+h/2])])

        dis_ = np.array([ ((one[1]-sec_center)**2).sum() for one in boxs])
        if len(dis_)==0:
            kernal = [0, 0, section.shape[1], section.shape[0]]
        else:
            kernal = boxs[dis_.argmin()][0]

        center_c  = (kernal[0]+kernal[2]/2,kernal[1]+kernal[3]/2)
        w_2 = int(width/2)
        h_2 = kernal[3]/2

        if center_c[0] - w_2< 0:
            w_2 = center_c[0]
        new_box = [center_c[0] - w_2,kernal[1],width,kernal[3]]
        # print new_box[2]/float(new_box[3])
        if new_box[2]/float(new_box[3])>0.5:
            # print "异常"
            h = int((new_box[2]/0.35 )/2)
            if h>35:
                h = 35
            new_box[1] = center_c[1]- h
            if new_box[1]<0:
                new_box[1] = 1
            new_box[3] = h*2
        section  = section[int(new_box[1]):int(new_box[1]+new_box[3]), int(new_box[0]):int(new_box[0]+new_box[2])]
        new_sections.append(section)
        # print new_box
    return new_sections
def slidingWindowsEval(image):
    windows_size = 16
    stride = 1
    #分割开始
    height= image.shape[0]
    data_sets = []
    for i in range(0,image.shape[1]-windows_size+1,stride):
        data = image[0:height,i:i+windows_size]
        data = cv2.resize(data,(23,23))
        data = cv2.equalizeHist(data)
        data = data.astype(np.float)/255
        data=  np.expand_dims(data,3)
        data_sets.append(data)
    res = model2.predict(np.array(data_sets)) #用模型预测
    pin = res
    p = 1 -  (res.T)[1]
    p = f.gaussian_filter1d(np.array(p,dtype=np.float),3)
    lmin = l.argrelmax(np.array(p),order = 3)[0]
    interval = []
    for i in range(len(lmin)-1):
        interval.append(lmin[i+1]-lmin[i])

    if(len(interval)>3):
        mid  = get_median(interval)
    else:
        return []
    pin = np.array(pin)
    res =  searchOptimalCuttingPoint(image,pin,0,mid,3)

    cutting_pts = res[1]
    last =  cutting_pts[-1] + mid
    if last < image.shape[1]:
        cutting_pts.append(last)
    else:
        cutting_pts.append(image.shape[1]-1)
    name = ""
    confidence =0.00
    seg_block = []
    for x in range(1,len(cutting_pts)):
        if x != len(cutting_pts)-1 and x!=1:
            section = image[0:36,cutting_pts[x-1]-2:cutting_pts[x]+2]
        elif  x==1:
            c_head = cutting_pts[x - 1]- 2
            if c_head<0:
                c_head=0
            c_tail = cutting_pts[x] + 2
            section = image[0:36, c_head:c_tail]
        elif x==len(cutting_pts)-1:
            end = cutting_pts[x]
            diff = image.shape[1]-end
            c_head = cutting_pts[x - 1]
            c_tail = cutting_pts[x]
            if diff<7 :
                section = image[0:36, c_head-5:c_tail+5]
            else:
                diff-=1
                section = image[0:36, c_head - diff:c_tail + diff]
        elif  x==2:
            section = image[0:36, cutting_pts[x - 1] - 3:cutting_pts[x-1]+ mid]
        else:
            section = image[0:36,cutting_pts[x-1]:cutting_pts[x]]
        seg_block.append(section)
    refined = refineCrop(seg_block,mid-1)
    #识别开始
    for i,one in enumerate(refined):
        res_pre = cRP.SimplePredict(one, i )
        confidence+=res_pre[0]
        name+= res_pre[1]
    return refined,name,confidence
