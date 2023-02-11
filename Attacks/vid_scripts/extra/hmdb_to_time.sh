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
    --src_num_cls 51 \
    --batch_size 7 \
    --num_frames 8 \
    --num_gpus 1 \
    --src_frames 8 \
    --eps 70 \
    --num_div_gpus 1 \
    --variation "hmdb" \
    --no_unsup_loss True 
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

for ATTACK in 'dim' 'mifgsm' 'fgsm' 'pgd'; do
  evaluation 'hmdb51' "resnet_50" "timesformer_vit_base_patch16_224" "" "$ATTACK" -1 10 16 "last" "pretrained_video_models/resnet_50/hmdb/8_joint_1p/results/checkpoint_epoch_00030.pyth" "pretrained_video_models/timesformer/hmdb/8_div/results/checkpoint_epoch_00015.pyth"
done
# evaluation 'hmdb51' "timesformer_vit_base_patch16_224" "resnet_50"  "" "dim" -1 20 16 "last" "pretrained_video_models/timesformer/ssv2/8_div.pyth" "pretrained_video_models/resnet_50/hmdb/8_joint_1p/results/checkpoint_epoch_00030.pyth"