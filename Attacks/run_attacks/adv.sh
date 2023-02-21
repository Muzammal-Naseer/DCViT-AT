#!/bin/bash

evaluation() {
  python eval_adv_base.py \
    --test_dir "path/to/annotation/file" \ # path to the annotation file
    --data_type "$1" \ # dataset name
    --src_model "$2" \ # source model name
    --tar_model "$3" \ # target model name
    --image_list "$4" \ 
    --attack_type "$5" \ # attack type
    --target_label "$6" \ # target label for targeted attacks (-1 for untargeted attacks)
    --iter "$7" \ # number of iterations
    --eps "$8" \ # epsilon (should be 16 for IN and 70 for videos because of normalization)
    --index "$9" \ # index of the frames to be attacked (last or all)
    --pre_trained "${10}" \ # path to the source model
    --tar_pre_trained "${11}" \ # path to the target model
    --num_temporal_views 3 \ # number of temporal views
    --num_classes 101 \ # number of classes in the target dataset
    --src_num_cls 101 \ # number of classes in the source dataset
    --batch_size 8 \ 
    --num_frames 8 \ # number of frames in the input of the target model
    --num_gpus 1 \ # number of GPUs - currently only 1 is supported
    --src_frames 8 \ # number of frames in the input of the source model
    --num_div_gpus 1 \ 
    --add_grad True \ # add the gradient of the main frame to all other frames
    --variation "20iter" 
}

for ATTACK in 'dim' 'mifgsm' 'pgd' 'fgsm'; do
  evaluation 'ucf101' "deit_base_patch16_224_timeP_1_cat" "resnet_50"  "" "$ATTACK" -1 20 70 "all" "path/to/source/model" "path/to/target/model"
done
