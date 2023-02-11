#!/bin/bash

DATA_DIR="../../../data"

evaluation() {
  python eval_adv_base.py \
    --test_dir "$DATA_DIR" \
    --data_type "$1" \
    --src_model "$2" \
    --tar_model "$3" \
    --image_list "$4" \
    --attack_type "$5" \
    --target_label "$6" \
    --iter "$7" \
    --eps "$8" \
    --index "$9"
}

evaluation_ens() {
  python eval_adv_res_ens.py \
    --test_dir "$DATA_DIR" \
    --data_type "$1" \
    --src_model "$2" \
    --tar_model "$3" \
    --image_list "$4" \
    --attack_type "$5" \
    --target_label "$6" \
    --iter "$7" \
    --eps "$8" \
    --index "$9" \
    --res "${10}"
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

evaluation 'IN' "deit_base_patch16_224" "resnet152"  "image_list_5k.json" "dim" -1 10 16 "all"

#evaluation 'IN' "deit_base_patch16_224" "T2t_vit_7"  "image_list_5k.json" "pifgsm" -1 10 16 "last"
#evaluation 'IN' "deit_base_patch16_224" "T2t_vit_7"  "image_list_5k.json" "pifgsm" -1 10 16 "all"

#evaluation_ens  'IN' "deit_base_patch16_224_resPT_1" "resnet152"  "image_list_5k.json" "mifgsm" -1 10 16 "all" "56_96_120_224"