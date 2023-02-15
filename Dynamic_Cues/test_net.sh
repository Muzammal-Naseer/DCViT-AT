python tools/run_net.py \
  --cfg configs/Kinetics/8_224_TEST.yaml \
  DATA.PATH_TO_DATA_DIR '../../../../data/drive_2/repos/datasets/kinetics-dataset/k400_resized/annotations_1500' \
  TRAIN.ENABLE False \
  TEST.CHECKPOINT_FILE_PATH ../../ImgModelsToAll/pretrained_video_models/timesformer/k400/8_div.pyth \
  MODEL.MODEL_NAME vit_base_patch16_224 \
  TEST.SAVE_RESULTS_PATH hmdb/time_k400_shuffle.pkl \
  DATA.TEST_CROP_SIZE 224 \
  DATA.NUM_FRAMES 8 \
  TEST.NUM_ENSEMBLE_VIEWS 3 \
  NUM_GPUS 4 \
  TEST.BATCH_SIZE 16 \
  TEST.NUM_SPATIAL_CROPS 3