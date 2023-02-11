python tools/run_net.py \
  --cfg configs/Ucf101/TimeSformer_divST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR '../../../../data/drive_2/ucf101/annotations_svt' \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 64 \
  MODEL.MODEL_NAME deit_tiny_patch16_224_timeP_1 \
  MODEL.NUM_CLASSES 101 \
  OUTPUT_DIR train_output/196_cat_prompt/ucf/deit/8_224_joint_1p/results/ \
  TIMESFORMER.ATTENTION_TYPE 'joint_space_time' \
  TRAIN.FINETUNE False \
  SOLVER.BASE_LR 0.005 \
  SOLVER.MAX_EPOCH 15 \
  TRAIN.EVAL_PERIOD 5 \
  DATA.NUM_FRAMES 16 \
  SOLVER.STEPS '[0,11,14]' \
  SOLVER.LRS '[1,0.1,0.01]' \
  TRAIN.CHECKPOINT_PERIOD 15 \
  DATA.TRAIN_JITTER_SCALES '[256,320]' \
  DATA.TRAIN_CROP_SIZE 224

# python tools/run_net.py \
#   --cfg configs/Hmdb51/TimeSformer_divST_8x32_224.yaml \
#   DATA.PATH_TO_DATA_DIR '../../../../data/drive_2/hmdb51' \
#   NUM_GPUS 4 \
#   TRAIN.BATCH_SIZE 32 \
#   MODEL.MODEL_NAME deit_base_patch16_224_timeP_1 \
#   MODEL.NUM_CLASSES 51 \
#   OUTPUT_DIR train_output/196_cat_prompt/hmdb/deit/8_4blocks \
#   TIMESFORMER.ATTENTION_TYPE 'joint_space_time' \
#   TRAIN.FINETUNE False \
#   SOLVER.BASE_LR 0.005 \
#   SOLVER.MAX_EPOCH 15 \
#   TRAIN.EVAL_PERIOD 5 \
#   DATA.NUM_FRAMES 8 \
#   SOLVER.STEPS '[0,11,14]' \
#   SOLVER.LRS '[1,0.1,0.01]' \
#   TRAIN.CHECKPOINT_PERIOD 15 \
#   DATA.TRAIN_JITTER_SCALES '[256,320]' \
#   DATA.TRAIN_CROP_SIZE 224
