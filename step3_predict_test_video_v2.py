#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/9/19 10:51 AM
# @Author  : Zhuang Yuzhou
# @File    : step3_predict_test_video_v2.py

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
from safety_helmet_config import safety_helmet_config
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

import datetime

def draw_chinese_warning_caption(draw_img, img_size, input_text='检测到管制刀具！', top_left_x=0, top_left_y=0 , color=(255, 0, 0),bias=15, font_size=25):
    frame_width, frame_height = img_size
    pil_img = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)  # Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)  # PIL图片上打印汉字
    font = ImageFont.truetype("./font_file/simhei.ttf", font_size, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小；Windows系统“simhei.ttf”默认存储在路径：C:\Windows\Fonts中
    draw.text((top_left_x+bias, top_left_y+bias), str(input_text), color, font=font)
    cv2img = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)  # 将图片转成cv2.imshow()可以显示的数组格式
    return cv2img

if __name__ == '__main__':
    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
    model_path = './model_save/resnet101_csv_15.h5'

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet101')

    # anchor settings
    # optionally load config parameters
    anchor_parameters = None
    anchor_config = safety_helmet_config.anchor_config_file
    if anchor_config:
        anchor_config = read_config_file(anchor_config)
        print('anchor_config', anchor_config)
        if 'anchor_parameters' in anchor_config:
            anchor_parameters = parse_anchor_parameters(anchor_config)
    print('anchor_parameters', anchor_parameters)

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet
    # converting-a-training-model-to-inference-model
    model = models.convert_model(model, anchor_params=anchor_parameters)
    print(model.summary())

    # load label to names mapping for visualization purposes
    labels_to_names = safety_helmet_config.labels_to_names

    # load vedio
    video_name = './video_source/安全帽_录制视频_2.mp4'
    # video_name = 0

    cap = cv2.VideoCapture(video_name)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # 读取视频的fps,  大小
    fps = cap.get(cv2.CAP_PROP_FPS)
    # # slow
    # fps = 20
    if fps < 1:
        fps = 24

    file_name = './安全帽_录制视频_2_detect.mp4'
    print('write name:%s,fps:%s,size:%s' % (file_name, fps, size))
    videoWriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

    count = 0
    while True:
        ret, image = cap.read()
        if ret == False:
            break
        print(image.shape)
        # copy to draw on
        draw = image.copy()

        # --------------------- 添加提示信息 ----------------------
        # 地点
        video_site_context = '工地1号'
        # 时间
        time_now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f'))
        video_time_now_context = time_now[:-4]
        # 报警信息
        video_safety_helmet_warning_context = '存在未佩戴安全帽的现象！'
        # 展示信息
        video_site_information = '地点: %s' % (video_site_context)
        video_time_information = '时间: %s' % (video_time_now_context)
        video_warning_information = '检测信息:'

        print(video_time_information)
        draw = draw_chinese_warning_caption(draw, img_size=size,
                                            input_text=video_site_information,
                                            top_left_y=0,
                                            color=(255, 0, 0))
        draw = draw_chinese_warning_caption(draw, img_size=size,
                                            input_text=video_time_information,
                                            top_left_y=40,
                                            color=(255, 0, 0))

        draw = draw_chinese_warning_caption(draw, img_size=size,
                                            input_text=video_warning_information,
                                            top_left_y=80,
                                            color=(255, 0, 0))
        # -------------------------- 添加完成 -------------------------------

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image, min_side=1000, max_side=1500)

        # process image
        start = time.time()
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.8:
                break

            self_label_to_color = {0: (0, 255, 0), 1: (0, 0, 255)}

            # color = label_color(label)
            color = self_label_to_color[label]

            b = box.astype(int)
            draw_box(draw, b, color=color)
            try:
                caption = "{} {:.3f}".format(labels_to_names[label], score)
            except:
                caption = ''
            draw_caption(draw, b, caption)

            # 添加信息
            if label == 1:
                video_warning_information = '检测信息: %s' % (video_safety_helmet_warning_context)
                draw = draw_chinese_warning_caption(draw, img_size=size,
                                                    input_text=video_warning_information,
                                                    top_left_y=80,
                                                    color=(255, 0, 0))

        # # 写视频帧
        videoWriter.write(draw)
        cv2.imshow('object detection', cv2.resize(draw, (1200, 600)))

        # calculate the frame per second

        count += 1
        if count == 10:
            # End time
            end = time.time()

            # Time elapsed
            seconds = end - start

            # Calculate frames per second
            fps = count / seconds

            count = 0

            print("---Frame Per Second---", fps)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
