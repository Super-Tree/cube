# coding=utf-8
import os.path as osp
from easydict import EasyDict as edict
from distutils import spawn
import random
import string
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0':log, '1':info, '2':warning ,'3':Error}
__C = edict()

cfg = __C

__C.GPU_AVAILABLE = '3,1,2,0'
__C.GPU_USE_COUNT = len(__C.GPU_AVAILABLE.split(','))
__C.GPU_MEMORY_FRACTION = 1
__C.VOXEL_POINT_COUNT = 35
__C.NUM_CLASS = 2
__C.DEFAULT_PADDING = 'SAME'
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))
__C.OUTPUT_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'output'))
__C.LOG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'log'))
__C.LOCAL_LOG_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'local_log'))
__C.TEST_RESULT = osp.abspath(osp.join(__C.ROOT_DIR, 'test_result'))
__C.EPS = 1e-15
__C.ANCHOR = [4.000,4.000,2.000]  # car size # todo: car height should be carefully decided!
__C.CUBIC_RES = [0.286,0.286,0.143]  # car size
__C.ANCHOR_CNT=1
__C.RPN_POINTS_REMAIN = 600
__C.DETECTION_RANGE = 45.
__C.RANDOM_STR =''.join(random.sample(string.ascii_letters, 4))
if spawn.find_executable("nvcc",path="/usr/local/cuda-8.0/bin/"):
    # Use GPU implementation of non-maximum suppression
    __C.USE_GPU_NMS = True

    # Default GPU device id
    __C.GPU_ID = 0
else:
    print ('File: config.py '
           'Notice: nvcc not found')
    __C.USE_GPU_NMS = False


# Training options
__C.TRAIN = edict()

__C.TRAIN.SNAPSHOT_ITERS = 5000
__C.TRAIN.LEARNING_RATE = 1e-5
__C.TRAIN.BATCH_SIZE = 1  # only one image
__C.TRAIN.FOCAL_LOSS = True
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.MOVING_AVERAGE_DECAY = 0.9999
__C.TRAIN.ITER_DISPLAY = 10
__C.TRAIN.TENSORBOARD = True
__C.TRAIN.DEBUG_TIMELINE = True  # Enable timeline generation
__C.TRAIN.USE_VALID = True
__C.TRAIN.VISUAL_VALID = True

# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 128
# Max number of foreground examples ,only keep 1/4 positive anchors
__C.TRAIN.RPN_FG_FRACTION = 0.25
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.75
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.4
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False


# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 4000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 50
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.5


# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.7
# Overlap threshold for a ROI to be considered background (class = 0 if overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1


# Testing options
__C.TEST = edict()
__C.TEST.ITER_DISPLAY = 1
__C.TEST.SAVE_IMAGE = True
# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.32

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 3000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 50
__C.TEST.TENSORBOARD = False
__C.TEST.DEBUG_TIMELINE = False
