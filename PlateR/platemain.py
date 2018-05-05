#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:piepis
@file:platemain.py.py
@time:2018-03-1410:23
@desc:
'''
import common
from  PlateCode import pipline as pp
import  cv2
# 定义函数，参数是函数的两个参数，都是python本身定义的，默认就行了。

img = cv2.imread('44.jpg')
result = pp.SimpleRecognizePlate(image=img) #图片结果
dic={ }
for platenum ,platestr in result:
    dic[platenum]=platestr




# class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
#     def handle(self):
#         print('New connection:', self.client_address)
#         while True:
#             data = self.request.recv(1024*1024)
#             # image = np.asarray(bytearray(data), dtype="uint8")
#             # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#             # # x = np.fromfile(data, dtype=np.ubyte)
#             # cv2.imwrite('tttttttt1111.jpg',image
#             if data:
#                 t0 = time.time()
#                 data = str(data,'utf-8')
#                 image = cv2.imread(data)
#                 t1 = time.time()-t0
#                 result =pp.SimpleRecognizePlate(image=image)
#                 t2 = time.time()-t1-t0
#                 # str = '字符串检验'
#                 if result:
#                     response = bytes(result[1], 'utf-8')
#                 else:
#                     response = '字符串检验'
#                     response = bytes(response, 'utf-8')
#                 self.request.sendall(response)
#                 t3 = time.time() - t2-t0-t1
#                 print('{0}  {1}   {2}'.format(t1,t2,t3))
#             else:
#                 break






