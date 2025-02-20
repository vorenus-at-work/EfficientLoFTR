from configs.data.base import cfg
# running bash scripts
TRAIN_BASE_PATH = "data/megadepth/index"

# running debug
#TRAIN_BASE_PATH = "EfficientLoFTR/data/megadepth/index"
cfg.DATASET.TRAINVAL_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TRAIN_DATA_ROOT = "data/megadepth/train"
# running debug
#cfg.DATASET.TRAIN_DATA_ROOT = "EfficientLoFTR/data/megadepth/train"
cfg.DATASET.TRAIN_NPZ_ROOT = f"{TRAIN_BASE_PATH}/scene_info_0.1_0.7"
cfg.DATASET.TRAIN_LIST_PATH = f"{TRAIN_BASE_PATH}/trainvaltest_list/train_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TRAIN = 0.0
# running bash scripts for MegaDepth dataset preparation
TEST_BASE_PATH = "data/megadepth/index"
# running debug 
#TEST_BASE_PATH = "EfficientLoFTR/data/megadepth/index"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
# running debug
#cfg.DATASET.VAL_DATA_ROOT = cfg.DATASET.TEST_DATA_ROOT = "EfficientLoFTR/data/megadepth/test"
cfg.DATASET.VAL_NPZ_ROOT = cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}/scene_info_val_1500"
cfg.DATASET.VAL_LIST_PATH = cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/trainvaltest_list/val_list.txt"
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0   # for both test and val

# 368 scenes in total for MegaDepth
# (with difficulty balanced (further split each scene to 3 sub-scenes))
cfg.TRAINER.N_SAMPLES_PER_SUBSET = 100
# changed by vorenus from 832 to 256
cfg.DATASET.MGDPT_IMG_RESIZE = 832  # for training on 32GB meme GPUs
#cfg.DATASET.MGDPT_IMG_RESIZE = 512  # for training in 4090 meme GPUs

cfg.DATASET.NPE_NAME = 'megadepth'