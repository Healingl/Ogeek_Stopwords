#!/usr/bin/env bash
nohup python keras_retinanet/bin/train.py --batch-size 1 --image-min-side 500 --image-max-side 700 --epochs 20 --backbone resnet101 --gpu 2 --snapshot-path ./model_save/ csv ./input_retinanet_data/train_knife_data_own_and_normal.csv ./input_retinanet_data/classes.csv --val-annotations ./input_retinanet_data/val_knife_data.csv >/root/7_31_knife_training.log &

nohup python keras_retinanet/bin/train.py --batch-size 3 --image-min-side 500 --image-max-side 700 --epochs 20 --backbone resnet101 --gpu 2,3 --multi-gpu 2 --multi-gpu-force --snapshot-path ./model_save/ csv ./input_retinanet_data/train_knife_data_own_and_normal.csv ./input_retinanet_data/classes.csv --val-annotations ./input_retinanet_data/val_knife_data.csv >/root/7_31_knife_training.log &