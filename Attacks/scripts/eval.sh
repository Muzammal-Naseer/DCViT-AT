#!/bin/bash

DATA_DIR="../../../data"

evaluation() {
  python eval.py \
    --test_dir "$DATA_DIR" \
    --data_type "$1" \
    --src_model "$2" \
    --image_list "$3" \
    --img_size "$4" \
    --pre_trained "$5" \
    --batch_size 10
}


for MODEL in "resnet152" "vgg19_bn" "deit_tiny_patch16_224" "deit_small_patch16_224" "deit_base_patch16_224" 'T2t_vit_7' 'T2t_vit_24' 'tnt_s_patch16_224' 'vit_base_patch16_224' 'BIT'; do
  for RES in 56 96 120 224; do
    evaluation "IN" "$MODEL" "image_list_50k.json"  "$RES" ""
  done
done