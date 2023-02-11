#!/bin/bash

DATA_DIR="../../../data/drive_2/ucf101/annotations_svt"

evaluation() {
  python eval.py \
    --test_dir "$DATA_DIR" \
    --data_type "$1" \
    --src_model "$2" \
    --img_size "$3" \
    --pre_trained "$4" \
    --batch_size 10 \
    --num_classes 101 \
    --num_temporal_views 3 \
    --num_frames 8
}


evaluation "ucf101" "deit_base_patch16_224_timeP_1_cat" "224" "pretrained_video_models/transformation/1_8split_cat_prompt/ucf/deit/8_224_joint_1p/results/checkpoint_epoch_00015.pyth" 