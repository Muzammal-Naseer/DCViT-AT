#!/bin/bash

# DATA_DIR = "../data/kinetics-dataset/annotations_4k"

evaluation() {
  python eval_adv_base.py \
    --test_dir "../../../data/drive_2/hmdb51_2" \
    --data_type "$1" \
    --src_model "$2" \
    --tar_model "$3" \
    --image_list "$4" \
    --attack_type "$5" \
    --target_label "$6" \
    --iter "$7" \
    --eps "$8" \
    --index "$9" \
    --pre_trained "${10}" \
    --tar_pre_trained "${11}" \
    --num_temporal_views 3 \
    --num_classes 51 \
    --src_num_cls 400 \
    --batch_size 6 \
    --num_frames 8 \
    --num_gpus 1 \
    --src_frames 8 \
    --eps 70 \
    --num_div_gpus 1 \
    --variation "k400" \
    --no_sup_loss True
}

#for ATTACK in 'mifgsm', 'dim'; do
#  for MODEL in "resnet152" "BIT" 'wide_resnet50_2' 'densenet201' 'T2t_vit_7' 'T2t_vit_24' 'tnt_s_patch16_224' 'vit_base_patch16_224' 'BIT'; do
#    for INDEX in  'last' 'all'; do
#      for RES in  '56' '96' '120' '224' '56_96' '56_96_120' '56_96_120_224'; do
#        evaluation_ens  'IN' "deit_base_patch16_224_resPT_1" "$MODEL"  "image_list_5k.json" "$ATTACK" -1 10 16 "$INDEX" "$RES"
#      done
#    done
#  done
#done

for ATTACK in 'dim'; do
  evaluation 'hmdb51' "timesformer_vit_base_patch16_224" "i3d" "" "$ATTACK" -1 10 16 "last" "pretrained_video_models/timesformer/k400/8_div.pyth" "pretrained_video_models/i3d/hmdb/checkpoint_epoch_00120.pyth"
done
# evaluation 'ucf101' "timesformer_vit_base_patch16_224" "resnet_50"  "" "dim" -1 20 16 "last" "pretrained_video_models/timesformer/ssv2/8_div.pyth" "pretrained_video_models/resnet_50/ucf/8_joint_1p/results/checkpoint_epoch_00030.pyth"

# for ATTACK in 'dim' 'mifgsm' 'pgd' 'fgsm'; do
#   evaluation 'ucf101' "deit_small_patch16_224_timeP_1" "resnet_50"  "" "$ATTACK" -1 20 16 "all" "pretrained_video_models/transformation/1_8split_cat_prompt/ucf/deit_small/8_joint_1p/results/checkpoint_epoch_00015.pyth" "pretrained_video_models/resnet_50/ucf/8_joint_1p/results/checkpoint_epoch_00030.pyth"
# done

# for ATTACK in 'dim' 'mifgsm' 'pgd' 'fgsm'; do
#   evaluation 'ucf101' "deit_tiny_patch16_224_timeP_1" "resnet_50"  "" "$ATTACK" -1 20 16 "all" "pretrained_video_models/transformation/1_8split_cat_prompt/ucf/deit_tiny/8_joint_1p/results/checkpoint_epoch_00015.pyth" "pretrained_video_models/resnet_50/ucf/8_joint_1p/results/checkpoint_epoch_00030.pyth"
# done

# evaluation 'kinetics' "deit_base_patch16_224_timeP_1_cat" "timesformer_vit_base_patch16_224"  "" "dim" -1 10 16 "all" "pretrained_video_models/transformation/k400/deit/8_joint_1p/results/checkpoint_epoch_00015.pyth" "pretrained_video_models/timesformer_main/k400/8_joint_1p/results/checkpoint_epoch_00015.pyth"

# pip install https://github.com/MadryLab/robustness/tarball/4033befe273b29f7b6dc36c30aa40696ed8fae96

# evaluation 'ucf101' "deit_base_patch16_224_timeP_1_cat" "timesformer_vit_base_patch16_224"  "" "dim" -1 10 16 "all" "pretrained_video_models/transformation/1_8split_cat_prompt/ucf/deit/8_224_joint_1p/results/checkpoint_epoch_00015.pyth" "pretrained_video_models/timesformer/ucf/8_224_joint/results/checkpoint_epoch_00015.pyth"

#evaluation 'IN' "deit_base_patch16_224" "T2t_vit_7"  "image_list_5k.json" "pifgsm" -1 10 16 "last"
#evaluation 'IN' "deit_base_patch16_224" "T2t_vit_7"  "image_list_5k.json" "pifgsm" -1 10 16 "all"

#evaluation_ens  'IN' "deit_base_patch16_224_resPT_1" "resnet152"  "image_list_5k.json" "mifgsm" -1 10 16 "all" "56_96_120_224"