# python tools/run_net.py \
#   --cfg configs/Hmdb51/TimeSformer_divST_8x32_224_TEST.yaml \
#   DATA.PATH_TO_DATA_DIR '../../../../data/drive_2/hmdb51' \
#   TRAIN.ENABLE False \
#   TEST.CHECKPOINT_FILE_PATH train_output/196_cat_prompt/hmdb/deit_base/8_joint_1p_96/results/checkpoint_epoch_00015.pyth \
#   MODEL.MODEL_NAME deit_base_patch16_224_timeP_1 \
#   TEST.SAVE_RESULTS_PATH hmdb/deit_96.pkl \
#   DATA.TEST_CROP_SIZE 96 \
#   DATA.NUM_FRAMES 8 \
#   TIMESFORMER.ATTENTION_TYPE 'joint_space_time' \
#   TEST.NUM_ENSEMBLE_VIEWS 3 \
#   NUM_GPUS 2 \
#   TEST.BATCH_SIZE 16 \
#   TEST.NUM_SPATIAL_CROPS 3
# python tools/run_net.py \
#   --cfg configs/Hmdb51/TimeSformer_divST_8x32_224_TEST.yaml \
#   DATA.PATH_TO_DATA_DIR '../../../../data/drive_2/hmdb51' \
#   TRAIN.ENABLE False \
#   TEST.CHECKPOINT_FILE_PATH train_output/196_cat_prompt/hmdb/deit/8_4blocks/results/checkpoint_epoch_00015.pyth \
#   MODEL.MODEL_NAME deit_base_patch16_224_timeP_1 \
#   TEST.SAVE_RESULTS_PATH hmdb/deit_4block.pkl \
#   DATA.TEST_CROP_SIZE 224 \
#   DATA.NUM_FRAMES 8 \
#   TIMESFORMER.ATTENTION_TYPE 'joint_space_time' \
#   TEST.NUM_ENSEMBLE_VIEWS 3 \
#   NUM_GPUS 2 \
#   TEST.BATCH_SIZE 16 \
#   TEST.NUM_SPATIAL_CROPS 3

python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224_TEST.yaml \
  DATA.PATH_TO_DATA_DIR '../../../../data/drive_2/repos/datasets/kinetics-dataset/k400_resized/annotations_1500' \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH ../../ImgModelsToAll/pretrained_video_models/timesformer/k400/8_div.pyth \
  MODEL.MODEL_NAME vit_base_patch16_224 \
  TEST.SAVE_RESULTS_PATH hmdb/time_k400_shuffle.pkl \
  DATA.TEST_CROP_SIZE 224 \
  DATA.NUM_FRAMES 8 \
  TIMESFORMER.ATTENTION_TYPE 'divided_space_time' \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  NUM_GPUS 4 \
  TEST.BATCH_SIZE 16 \
  TEST.NUM_SPATIAL_CROPS 3