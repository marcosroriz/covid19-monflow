@echo off
set list=centernet_hg104_1024x1024_coco17_tpu-32 efficientdet_d4_coco17_tpu-32 ssd_resnet50_v1_fpn_1024x1024_coco17_tpu-8 efficientdet_d7_coco17_tpu-32 ssd_resnet101_v1_fpn_1024x1024_coco17_tpu-8 ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8 faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8 faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8 faster_rcnn_inception_resnet_v2_1024x1024_coco17_tpu-8 faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8 
(for %%a in (%list%) do ( 
   python main_process_raw.py --inputdir=C:/Imagens/T/ --model=%%a 
   python main_process_tiling.py --inputdir=C:/Imagens/T/ --model=%%a 
))
