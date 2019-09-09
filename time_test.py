#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 10:55
# @Author  : Zhuang Yuzhou
# @File    : test_knife_warning_caption.py

import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

from PIL import Image, ImageDraw, ImageFont

import datetime

def draw_chinese_warning_caption(draw_img, img_size, input_text='检测到管制刀具！', top_left_x=0, top_left_y=0 , color=(255, 0, 0),bias=10, font_size=30):
    frame_width, frame_height = img_size
    pil_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./font_file/simhei.ttf", font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小；Windows系统“simhei.ttf”默认存储在路径：C:\Windows\Fonts中
    draw.text((top_left_x+bias, top_left_y+bias), str(input_text), color, font=font)
    cv2img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式
    return cv2img
    

if __name__ == '__main__':
    
    # load vedio
    # video_name = './video_source/刀具检测_自制视频_1.mp4'
    video_name = 0
    
    cap = cv2.VideoCapture(video_name)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # 读取视频的fps,  大小
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 1:
        fps = 24
    
    # file_name = './自录制_挥刀视频_detect.mp4'
    # print('write name:%s,fps:%s,size:%s'%(file_name,fps,size))
    # videoWriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), fps, size)
    
    count = 0
    while True:
        ret, image = cap.read()
        if ret == False:
            break
        print(image.shape)
        
        # copy to draw on
        draw = image.copy()
        # 地点
        video_site_context = '天一国际广场-卡口3'
        # 时间
        time_now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f'))
        video_time_now_context = time_now[:-4]
        # 报警信息
        video_knife_warning_context = '检测到管制刀具！'
        
        # 展示信息
        video_site_information = '地点: %s'%(video_site_context)
        video_time_information = '时间: %s'%(video_time_now_context)
        video_warning_information = '检测信息: %s'%(video_knife_warning_context)
        
        
        draw = draw_chinese_warning_caption(draw, img_size=size, input_text=video_site_information,top_left_y=0, color=(255, 0, 0))
        draw = draw_chinese_warning_caption(draw, img_size=size, input_text=video_time_information,top_left_y=40, color=(255, 0, 0))
        draw = draw_chinese_warning_caption(draw, img_size=size, input_text=video_warning_information, top_left_y=80,color=(255, 0, 0))
        
        cv2.imshow('object detection', cv2.resize(draw, (700, 500)))
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
    # videoWriter.release()
    cv2.destroyAllWindows()
