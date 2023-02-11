import numpy as np
import os
import pickle
import torch
from fvcore.common.file_io import PathManager
import cv2
from einops import rearrange, reduce, repeat
import scipy.io
import sys

import imtt.utils.checkpoint as cu
import imtt.utils.distributed as du
import imtt.utils.logging as logging
import imtt.utils.misc as misc
import imtt.visualization.tensorboard_vis as tb
from imtt.datasets import loader
from imtt.models import build_model
from imtt.utils.meters import TestMeter

def acc1(preds, labels):
    count = 0
    for i in range(len(preds)):
        if torch.argmax(preds[i]) == labels[i]:
            count += 1
    
    return (count / len(labels)) * 100

def acc5(preds, labels):
    count = 0
    for i in range(len(preds)):
        top5 = torch.topk(preds[i], 5).indices
        if labels[i] in top5:
            count += 1
    
    return (count / len(labels)) * 100


if __name__ == "__main__":
    with open(str(sys.argv[1]), 'rb') as f:
        data = pickle.load(f)
        accur1 = acc1(data[0],data[1])
        print("top-1: ", accur1)
        accur5 = acc5(data[0],data[1])
        print("top-5: ", accur5)
    