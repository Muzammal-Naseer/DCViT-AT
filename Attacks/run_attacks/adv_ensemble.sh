#!/bin/bash

evaluation() {
  python eval_ens.py \
    --test_dir "path/to/annotation/file" \ # path to the annotation file
    --data_type "$1" \ # dataset name
    --src_model_1 "$2" \ # source model 1 name
    --tar_model "$3" \ # target model name
    --image_list "$4" \ 
    --attack_type "$5" \ # attack type
    --target_label "$6" \ # target label for targeted attacks (-1 for untargeted attacks)
    --iter "$7" \ # number of iterations
    --eps "$8" \ # epsilon (should be 16 for IN and 70 for videos because of normalization)
    --index "$9" \ # index of the frames to be attacked (last or all)
    --pre_trained_1 "${10}" \ # path to the source model 1
    --tar_pre_trained "${11}" \ # path to the target model
    --src_model_2 "${12}" \ # source model 2 name
    --src_model_3 "${13}" \ # source model 3 name
    --pre_trained_2 "${14}" \ # path to the source model 2
    --pre_trained_3 "${15}" \ # path to the source model 3
    --num_temporal_views 3 \ # number of temporal views
    --num_classes 51 \ # number of classes in the target dataset
    --batch_size 16 \ 
    --num_frames 8 \ # number of frames in the input of the target model
    --num_gpus 1 \ # number of GPUs - currently only 1 is supported
    --src_frames 8 \ # number of frames in the input of the source model
    --num_div_gpus 1 \ 
    --add_grad True \ # add the gradient of the main frame to all other frames
    --variation "ens" 
}

evaluation 'hmdb51' "deit_base_patch16_224_timeP_1_cat" "resnet_50" "" "dim" -1 20 70 "all" "path/to/source/model1" "path/to/target/model" "deit_small_patch16_224_timeP_1" "deit_tiny_patch16_224_timeP_1" "path/to/source/model2" "path/to/source/model3"