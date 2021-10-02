@echo off
set list=centernet_hg104_1024x1024_coco17_tpu-32 efficientdet_d4_coco17_tpu-32 
(for %%a in (%list%) do ( 
   python main_process_raw.py --inputdir=C:/Imagens/T/ --model=%%a 
   python main_process_tiling.py --inputdir=C:/Imagens/T/ --model=%%a 
))
