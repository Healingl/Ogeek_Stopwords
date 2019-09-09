#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 11:39
# @Author  : Zhuang Yuzhou
# @File    : time_test.py
import datetime


while True:
    time_now = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f'))
    time_now = time_now[:-4]
    
    
    print(time_now)
