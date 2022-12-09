CUDA_VISIBLE_DEVICES=0,1,2 python main_2cams_group.py \
 --savedir network001 \
 --model 'network' \
 --num-epoch 500 \
 --batch-size 24 \
 --loss_type 'focal' \
 --crop_width 768 \
 --crop_height 256 \
 --num_classes 8 \
 --ignore_class -1 \
 --datadir1 '/home/tekken/dataset/KITTI360/semantic_data' \
 --resize_width 1408 \
 --init_learningrate 0.5e-3 \
 --alpha 0.25 \
 --gamma 2.0 \



 
