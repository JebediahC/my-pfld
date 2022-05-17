"""
配置文件
"""

import os

# ------Tensorbaard------
TENSORBOARD = 'logs/tensorboard_logs'

# ------预处理-----------
DEBUG = False
MIRROR_FILE = "Mirror98.txt"
IMAGE_DIRS = 'WFLW_images'
LANDMARK_FILES = {"test":'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt', "train":'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'}
DATA_DIR = "data"
OUT_DIRS = {"test":"test_data", "train":"train_data"}
DATA_RENAME_PREFIX = "_"
TRAIN_SAMPLE_NUM = -1
TEST_SAMPLE_NUM = -1

# -----训练---------
# general
WORKERS = 0
DEVICES_ID = 0
TEST_INITIAL = 'False'
# optimizer
BASE_LR = 0.0001
WEIGHT_DECAY = 1e-6
# lr
LR_PATIENCE = 40
# epoch
START_EPOCH = 1
END_EPOCH = 300
SNAPSHOT = './snapshot/'
LOG_FILE = 'logs/train.logs'
RESUME = ''
# visualization
SHOW_ORIGINAL_IMAGE = 'True'
SHOW_BEFORE_TRAIN = 'True'
SHOW_EACH_EPOCH = 'True'
SHOW_TEST = 'True'
# dataset
DATA_ROOT = os.path.join(DATA_DIR, OUT_DIRS['train'], 'list.txt')
VAL_DATAROOT = os.path.join(DATA_DIR, OUT_DIRS['test'], 'list.txt')
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32

# pack logs
MILESTONES = [1,2,2,20,40,60,80,100,120,140,160,200]

# -------测试---------

MODEL_PATH = './ready_to_run/kaggle_checkpoint_epoch_125.pth.tar'
DEVICE = 'cpu'
VIDEO_SOURCE = 'my_v.mp4'