python tools/run_net.py \
  --cfg configs/Ucf101/8_224.yaml \
  DATA.PATH_TO_DATA_DIR '../../../../data/drive_2/ucf101/annotations_svt' \
  NUM_GPUS 2 \
  TRAIN.BATCH_SIZE 16 \
  MODEL.MODEL_NAME deit_base_patch16_224_timeP_1 \
  MODEL.NUM_CLASSES 101 \
  OUTPUT_DIR train_output/196_cat_prompt/test/depth \
  TRAIN.FINETUNE False \
  SOLVER.BASE_LR 0.005 \
  SOLVER.MAX_EPOCH 15 \
  TRAIN.EVAL_PERIOD 5 \
  DATA.NUM_FRAMES 8 \
  SOLVER.STEPS '[0,11,14]' \
  SOLVER.LRS '[1,0.1,0.01]' \
  TRAIN.CHECKPOINT_PERIOD 15 \
  DATA.TRAIN_JITTER_SCALES '[256,320]' \
  DATA.TRAIN_CROP_SIZE 224