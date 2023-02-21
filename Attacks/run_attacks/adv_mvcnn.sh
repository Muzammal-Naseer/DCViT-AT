#!/bin/bash

evaluation() {
  python eval_adv_mvcnn.py \
    --test_dir "/multi_modal_data/Depth/*/test" \ # path to test data
    --data_type "$1" \  # dataset name
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
    --num_temporal_views 3 \
    --num_classes 40 \ # number of classes in the target dataset
    --batch_size 8 \ 
    --num_frames 8 \ # number of frames in the input of the target model
    --num_gpus 1 \ # number of GPUs - currently only 1 is supported
    --src_frames 1 \ # number of frames in the input of the source model
    --num_div_gpus 1 \
    --no_sup_loss True \ # no supervised loss
    --variation "comp"
}

for ATTACK in 'fgsm' 'pgd' 'mifgsm'; do
  evaluation 'img3d_depth' "deit_base_image" "mvcnn" "" "$ATTACK" -1 20 16 "all" "$None" "pretrained_video_models/mvcnn/mvcnn_depth/model-00029.pth"
done

