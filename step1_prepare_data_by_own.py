#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/4 16:47
# @Author  : Zhuang Yuzhou
# @File    : step1_prepare_data.py
import pandas as pd
import json
import os
from tqdm import tqdm
from sklearn.utils import shuffle

def get_csv_dataset():
    knife_dataset_dir = 'D:/CETC_PROJECT_DATASET/daoju/outputs/'
    save_csv_dir = './keras_retinanet/CSV/'
    knife_file_name_list = os.listdir(knife_dataset_dir)
    print(len(knife_file_name_list))
    
    # columns_list: path, 左上角x, 左上角y, 右下角x, 右下角y, 类别
    annotations_all_data_list = []
    labeled_picture_num = 0
    for idx, knife_file_name in enumerate(tqdm(knife_file_name_list)):
        current_knife_json = json.load(open(knife_dataset_dir+knife_file_name, 'r', encoding='utf-8'))
        image_file_path = str(current_knife_json['path'])
        outputs = current_knife_json['outputs']
        if current_knife_json['labeled']:
            outputs_object_list = list(outputs['object'])
            labeled_picture_num += 1
            for outputs_object in outputs_object_list:
                current_object_class = str(outputs_object['name'])
                current_object_bndbox = outputs_object['bndbox']
                if current_object_class == '刀具':
                    annotations_all_data_list.append((image_file_path, current_object_bndbox['xmin'], current_object_bndbox['ymin'], current_object_bndbox['xmax'], current_object_bndbox['ymax'], 'knife'))
    
        # # 测试
        # print('current_knife_json',current_knife_json)
        # print('outputs',outputs)
        # print('outputs_object_list',outputs_object_list)
        # print('current all data')
        # print('annotations_all_data_list',annotations_all_data_list)
        
    
    print('all labeled picture num', labeled_picture_num)
    print('all annotations data num', len(annotations_all_data_list))
    
    # 划分训练集和验证集
    random_state = 2019
    split_percent = 0.8
    train_all_num = len(annotations_all_data_list)
    local_train_num = int(train_all_num * split_percent)
    shuffle_all_ann_data_list = shuffle(annotations_all_data_list, random_state=random_state)

    # 获得训练，验证的数据
    local_train_data_list = shuffle_all_ann_data_list[:local_train_num]
    local_val_data_list = shuffle_all_ann_data_list[local_train_num:]

    train_df = pd.DataFrame(columns=["path", "xmin", "ymin", "xmax", "ymax", "class"], data=local_train_data_list)
    val_df = pd.DataFrame(columns=["path", "xmin", "ymin", "xmax", "ymax", "class"], data=local_val_data_list)
    
    train_df.to_csv(save_csv_dir+'train_annotations.csv', index=False, header=False)
    val_df.to_csv(save_csv_dir+'val_annotations.csv', index=False, header=False)

    print('train annotations data num', len(train_df))
    print('val annotations data num', len(val_df))

def get_class_csv():
    save_csv_dir = './keras_retinanet/CSV/'
    class_id_list = [['knife', 0]]
    class_id_df = pd.DataFrame(columns=["class_name", "id"], data=class_id_list)
    class_id_df.to_csv(save_csv_dir+'classes.csv', index=False, header=False)
    
if __name__ == '__main__':
    get_csv_dataset()
    # get_class_csv()