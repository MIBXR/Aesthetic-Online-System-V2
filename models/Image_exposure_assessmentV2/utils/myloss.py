import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as ssim
import cv2
import torch.nn.functional as F
import os
import requests


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1Loss = nn.L1Loss(reduction='mean')

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape

        return self.L1Loss(p_target,p_estimate)