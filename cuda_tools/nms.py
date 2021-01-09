import torch
import numpy as np
from cuda_tools import ops

class NMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx, bboxes, iou_threshold):
        inds = ops.nms(bboxes, iou_threshold=float(iou_threshold))

        return inds

def cuda_nms(boxes, iou_threshold):
    inds = NMSop.apply(boxes, iou_threshold)
    return inds