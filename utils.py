import os
import json
import math
import torch
import numpy as np
import config as cfg
import torch.distributed as dist

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from cuda_tools import cuda_nms

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def xyxy2xywh(xyxy):
    """
    将xyxy格式的矩形框转化为xywh格式
    :param xyxy: shape为n*4, 即左上角和右下角
    :return: xywh: shape为n*4, 即左上角和wh
    """
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]

    w = x2 - x1
    h = y2 - y1

    xywh = torch.stack([x1,y1,w,h],dim=1)
    return xywh

def get_dist_info():
    assert dist.is_initialized(), "还没有初始化分布式!"
    assert dist.is_available(),"分布式在当前设备不可用!"
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    return rank,world_size

def reduce_mean(tensor):
    if not(dist.is_available() and dist.is_initialized()):
        return tensor
    tensor=tensor.clone()
    num_gpus = dist.get_world_size()
    num_gpus = float(num_gpus)
    tensor=tensor/num_gpus
    dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
    return tensor

def compute_iou_ltrb(ltrb1,ltrb2):
    """
    计算iou，但是输入的矩形框的格式为ltrb，即要求ltrb1和lrtb2相对应的矩形框是同心矩形框
    :param ltrb1:
    :param ltrb2:
    :return:
    """
    inter_ltrb=torch.min(ltrb1,ltrb2)
    inter_area=(inter_ltrb[:,0]+inter_ltrb[:,2])*(inter_ltrb[:,1]+inter_ltrb[:,3])
    area1=(ltrb1[:,0]+ltrb1[:,2])*(ltrb1[:,1]+ltrb1[:,3])
    area2=(ltrb2[:,0]+ltrb2[:,2])*(ltrb2[:,1]+ltrb2[:,3])
    union_area=area1+area2-inter_area
    union_area.clamp_(min=1e-6)
    ious=inter_area/union_area
    return ious

def compute_iou_xyxy(bboxes1,bboxes2):
    bboxes1 = bboxes1[:, None]
    bboxes2 = bboxes2[None, :]
    left = torch.max(bboxes1[..., 0], bboxes2[..., 0])
    top = torch.max(bboxes1[..., 1], bboxes2[..., 1])
    right = torch.min(bboxes1[..., 2], bboxes2[..., 2])
    bottem = torch.min(bboxes1[..., 3], bboxes2[..., 3])

    inter_width = (right - left).clamp(min=0)
    inter_height = (bottem - top).clamp(min=0)
    inter_area = inter_height * inter_width

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    union_area = area1 + area2 - inter_area
    union_area.clamp_(min=1e-6)
    ious = inter_area / union_area

    return ious

def synchronize():
    """启用分布式训练时，用于各个进程之间的同步"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def compute_base_anchors():
    base_anchors=[]
    for stride in cfg.fpn_strides:
        ratios = torch.tensor(cfg.anchor_ratio,dtype=torch.float).view(3,1)
        num_anchors = len(ratios)
        base_scale = torch.full(size=(num_anchors,2),fill_value=stride*cfg.anchor_scale,dtype=torch.float)
        ratios = torch.cat([ratios,1/ratios],dim=1)
        ratios = torch.sqrt(ratios)

        anchor_wh=base_scale*ratios
        anchor_start=torch.zeros_like(anchor_wh)
        anchor=torch.cat([anchor_start,anchor_wh],dim=1)
        anchor_center=(anchor_wh+anchor_start)*0.5
        anchor[:,0::2]-=anchor_center[:,[0]]
        anchor[:,1::2]-=anchor_center[:,[1]]
        base_anchors.append(anchor)
    return base_anchors

def compute_anchors(base_anchors,scales,device,dtype):
    all_anchors=[]
    for i,scale in enumerate(scales):
        stride=cfg.fpn_strides[i]
        y = torch.arange(scale[0],device=device,dtype=dtype)
        x = torch.arange(scale[1],device=device,dtype=dtype)
        y,x = torch.meshgrid(y,x)
        y = y.reshape(-1)
        x = x.reshape(-1)

        point = torch.stack([x,y,x,y],dim=-1)*stride

        base_anchors_level=base_anchors[i].view(1,4)
        base_anchors_level=base_anchors_level.type(dtype)
        base_anchors_level=base_anchors_level.to(device)

        anchors=point[:,None,:]+base_anchors_level[None,:,:]
        anchors=anchors.view(-1,4)

        all_anchors.append(anchors)
    return all_anchors

def compute_valid_flag(pad_img_shape, scales, device):
    valid_flag_per_img=[]
    for i, scale in enumerate(scales):
        stride=float(cfg.fpn_strides[i])
        h_fpn = scale[0]
        w_fpn = scale[1]
        h_valid = math.ceil(pad_img_shape[0]/stride)
        w_valid = math.ceil(pad_img_shape[1]/stride)

        y_valid = torch.zeros((h_fpn,), device=device, dtype=torch.bool)
        x_valid = torch.zeros((w_fpn,), device=device, dtype=torch.bool)
        x_valid[:w_valid] = 1
        y_valid[:h_valid] = 1

        y_valid,x_valid = torch.meshgrid(y_valid,x_valid)
        y_valid=y_valid.reshape(-1)
        x_valid=x_valid.reshape(-1)
        valid_flag_per_level=y_valid&x_valid

        valid_flag_per_img.append(valid_flag_per_level)
    return valid_flag_per_img

def reg_encode(anchors,gtbboxes,mean,std):
    ax=(anchors[:,0]+anchors[:,2])*0.5
    ay=(anchors[:,1]+anchors[:,3])*0.5
    aw=anchors[:,2]-anchors[:,0]
    ah=anchors[:,3]-anchors[:,1]

    gx=(gtbboxes[:,0]+gtbboxes[:,2])*0.5
    gy=(gtbboxes[:,1]+gtbboxes[:,3])*0.5
    gw=gtbboxes[:,2]-gtbboxes[:,0]
    gh=gtbboxes[:,3]-gtbboxes[:,1]

    dx=(gx-ax)/aw
    dy=(gy-ay)/ah
    dw=torch.log(gw/aw)
    dh=torch.log(gh/ah)
    delta=torch.stack([dx,dy,dw,dh],dim=-1)

    delta=delta.sub_(mean).div_(std)
    return delta

def reg_decode(anchors, delta,mean,std,max_shape=None):
    ax = (anchors[:, 0] + anchors[:, 2]) * 0.5
    ay = (anchors[:, 1] + anchors[:, 3]) * 0.5
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]

    delta=delta.mul_(std).add_(mean)

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]
    max_ratio=np.abs(np.log(cfg.decode_ratio_clip))
    dw = dw.clamp(min=-max_ratio,max=max_ratio)
    dh = dh.clamp(min=-max_ratio,max=max_ratio)

    predict_x = dx*aw+ax
    predict_y = dy*ah+ay
    predict_w = aw*torch.exp(dw)
    predict_h = ah*torch.exp(dh)
    x1 = predict_x - predict_w * 0.5
    y1 = predict_y - predict_h * 0.5
    x2 = predict_x + predict_w * 0.5
    y2 = predict_y + predict_h * 0.5
    if(max_shape is not None):
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1,y1,x2,y2],dim=-1)

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def load_model(model, epoch):
    """
    加载指定的checkpoint文件
    :param model:
    :param epoch:
    :return:
    """
    archive_path = os.path.join(cfg.archive_path, cfg.check_prefix+"_"+str(epoch)+".pth")
    check_point = torch.load(archive_path)
    state_dict = check_point["state_dict"]
    model.load_state_dict(state_dict)

def save_model(model,epoch):
    """保存训练好的模型，同时需要保存当前的epoch"""
    if(hasattr(model,"module")):
        model=model.module
    model_state_dict=model.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()
    checkpoint=dict(state_dict=model_state_dict,epoch=epoch)
    mkdir(cfg.archive_path)
    checkpoint_name=cfg.check_prefix+"_"+str(epoch)+".pth"
    checkpoint_path=os.path.join(cfg.archive_path,checkpoint_name)

    torch.save(checkpoint,checkpoint_path)

def mc_nms(scores, bboxes, labels, nms_thresholds):
    """
    对一张图片的原始预测结果进行nms,输入的scores是多个类别
    :param scores: 分类得分，shape为[N, ], 这里的N为初步筛选出的预测框数量
    :param bboxes: 预测框坐标，shape为[N, 4]。
    :param labels: 预测框类别，shape为[N, ]。
    :param nms_thresholds: nms中的筛选阈值，默认为0.5
    :return:
    """
    if bboxes.numel() == 0:
        bboxes = bboxes.new_zeros((0, 4))
        labels = bboxes.new_zeros((0, ), dtype=torch.long)
        scores = bboxes.new_zeros((0,))
        return scores, bboxes,labels

    max_coordinate=bboxes.max()
    offsets=labels*(max_coordinate+torch.tensor(1).to(bboxes))
    bboxes_for_nms=bboxes+offsets[:,None]

    _,inds = scores.sort(descending=True)
    scores = scores[inds]
    labels = labels[inds]
    bboxes = bboxes[inds]
    bboxes_for_nms=bboxes_for_nms[inds]

    inds = cuda_nms(bboxes_for_nms, iou_threshold=nms_thresholds)

    scores=scores[inds]
    bboxes=bboxes[inds]
    labels=labels[inds]

    return scores, bboxes, labels

def ml_nms(scores,bboxes,levels,nms_thresholds):
    max_coordinate=bboxes.max()
    offsets=levels*(max_coordinate+torch.tensor(1).to(bboxes))
    bboxes_for_nms=bboxes+offsets[:,None]

    _, inds = torch.sort(scores, descending=True)  # 首先根据score降序对bboxes和labels进行排列
    bboxes = bboxes[inds]
    scores = scores[inds]
    bboxes_for_nms = bboxes_for_nms[inds]

    inds = cuda_nms(bboxes_for_nms, iou_threshold=nms_thresholds)
    scores=scores[inds]
    bboxes=bboxes[inds]
    return scores, bboxes

def evaluate_coco(coco_gt, coco_results, output_path, iou_type="bbox"):
    with open(output_path, "w") as f:
        json.dump(coco_results, f)

    coco_dt = coco_gt.loadRes(output_path) if coco_results else COCO()
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval

"""计量时间和loss的工具"""
class AverageMeter():
    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, ncount=1):
        self.val=val
        self.sum+=val*ncount
        self.count+=ncount
        self.avg=self.sum/self.count