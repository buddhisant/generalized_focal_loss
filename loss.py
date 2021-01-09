import torch
import utils
import math
import config as cfg
import numpy as np
from torch import nn

import torch.nn.functional as F

INF = 100000000

class QFLLoss(nn.Module):
    def __init__(self):
        super(QFLLoss, self).__init__()
        self.weight=cfg.qfl_loss_weight
        self.beta=cfg.qfl_loss_beta

    def forward(self,preds,targets,avg_factor=None):
        loss=F.binary_cross_entropy_with_logits(preds,targets,reduction='none')
        preds_sigmoid=preds.sigmoid()
        scale_factor=(preds_sigmoid-targets).abs().pow(self.beta)
        loss=loss*scale_factor
        loss=loss.sum()*self.weight
        if(not avg_factor is None):
            return loss/avg_factor
        return loss

class GiouLoss(torch.nn.Module):
    def __init__(self):
        super(GiouLoss, self).__init__()
        self.weight = cfg.giou_loss_weight

    def forward(self,preds,targets,weights=None,avg_factor=None):
        """
        这里的preds和targets都是ltrb格式的
        :param preds:
        :param targets:
        :param weights:
        :return:
        """
        inter_ltrb = torch.min(preds, targets)
        inter_area = (inter_ltrb[:, 0] + inter_ltrb[:, 2]) * (inter_ltrb[:, 1] + inter_ltrb[:, 3])
        pred_area = (preds[:, 0] + preds[:, 2]) * (preds[:, 1] + preds[:, 3])
        target_area = (targets[:, 0] + targets[:, 2]) * (targets[:, 1] + targets[:, 3])
        union_area = pred_area + target_area - inter_area
        union_area.clamp_(min=1e-6)
        ious = inter_area / union_area

        enclose_ltrb = torch.max(preds,targets)
        enclose_area = (enclose_ltrb[:,0]+enclose_ltrb[:,2])*(enclose_ltrb[:,1]+enclose_ltrb[:,3])
        enclose_area.clamp_(min=1e-6)

        gious = ious - (enclose_area - union_area) / enclose_area
        loss = 1 - gious
        if(weights is not None):
            loss = loss*weights
        loss=self.weight*loss.sum()
        if (not avg_factor is None):
            return loss / avg_factor
        return loss

class DFLLoss(torch.nn.Module):
    def __init__(self):
        super(DFLLoss, self).__init__()
        self.weight=cfg.dfl_loss_weight

    def forward(self,preds,targets,weights=None,avg_factor=None):
        dis_left = targets.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - targets
        weight_right = targets - dis_left.float()
        loss = F.cross_entropy(preds, dis_left, reduction='none') * weight_left + F.cross_entropy(preds, dis_right, reduction='none') * weight_right
        if (not weights is None):
            loss=loss*weights
        loss=loss.sum()/4
        if (not avg_factor is None):
            return (loss / avg_factor)*self.weight
        return loss*self.weight

class Intergral(torch.nn.Module):
    def __init__(self):
        super(Intergral, self).__init__()
        self.reg_max=cfg.reg_max
        self.register_buffer("project",torch.linspace(0,self.reg_max,self.reg_max+1))

    def forward(self,x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x

class GFLLoss(torch.nn.Module):
    def __init__(self):
        super(GFLLoss, self).__init__()
        self.base_anchors = torch.tensor(cfg.base_anchors,dtype=torch.float)
        self.qfl_loss_func = QFLLoss()
        self.giou_loss_func = GiouLoss()
        self.dfl_loss_func = DFLLoss()
        self.num_classes=cfg.num_classes
        self.intergral=Intergral()
        self.reg_max=cfg.reg_max

    def collect_targets(self,pos_inds,neg_inds,assigned_gt_inds,gt_bbox,gt_label,anchors,valid,strides):
        target_per_img={}
        target_per_img["valid"]=valid
        target_per_img["pos_inds"]=pos_inds
        target_per_img["neg_inds"]=neg_inds

        valid_anchors=anchors[valid]
        valid_strides=strides[valid]
        pos_anchors=valid_anchors[pos_inds]
        pos_strides=valid_strides[pos_inds]

        pos_gt_bbox=gt_bbox[assigned_gt_inds,:]/pos_strides[:,None]

        pos_anchors_cx = ((pos_anchors[:, 0] + pos_anchors[:, 2]) * 0.5)/pos_strides
        pos_anchors_cy = ((pos_anchors[:, 1] + pos_anchors[:, 3]) * 0.5)/pos_strides

        pos_l = pos_anchors_cx - pos_gt_bbox[:, 0]
        pos_t = pos_anchors_cy - pos_gt_bbox[:, 1]
        pos_r = pos_gt_bbox[:, 2] - pos_anchors_cx
        pos_b = pos_gt_bbox[:, 3] - pos_anchors_cy

        target_per_img["pos_reg_targets"]=torch.stack([pos_l,pos_t,pos_r,pos_b],dim=-1)

        target_per_img["pos_cls_labels"]=gt_label[assigned_gt_inds]
        return target_per_img

    def compute_targets(self, anchors, valids, gt_bboxes, gt_labels, strides):
        targets=[]

        for i, gt_bbox in enumerate(gt_bboxes):
            gt_label=gt_labels[i]
            target_per_img={}

            num_gt=gt_bbox.size(0)
            valid_per_img=valids[i]
            num_valid_per_level=[valid_per_level.int().sum().item() for valid_per_level in valid_per_img]
            valid_per_img=torch.cat(valid_per_img)
            target_per_img["valid"]=valid_per_img

            valid_anchors_per_img=anchors[valid_per_img]
            num_valid_per_img=valid_anchors_per_img.size(0)

            anchors_cx = (valid_anchors_per_img[:,0]+valid_anchors_per_img[:,2])*0.5
            anchors_cy = (valid_anchors_per_img[:,1]+valid_anchors_per_img[:,3])*0.5
            anchors_center = torch.stack([anchors_cx,anchors_cy],dim=-1)

            gt_bbox_cx = (gt_bbox[:,0]+gt_bbox[:,2])*0.5
            gt_bbox_cy = (gt_bbox[:,1]+gt_bbox[:,3])*0.5
            gt_bbox_center = torch.stack([gt_bbox_cx,gt_bbox_cy],dim=-1)

            distance=(anchors_center[:,None,:]-gt_bbox_center[None,:,:]).pow(2).sum(-1).sqrt()
            distance=torch.split(distance,num_valid_per_level,dim=0)

            candidate_ids=[]
            start_idx=0
            for distance_per_level in distance:
                num_selected_cur_level=min(distance_per_level.size(0),cfg.num_candidate_per_level)
                _, candidate_ids_cur_level=distance_per_level.topk(num_selected_cur_level,dim=0,largest=False)
                candidate_ids.append(candidate_ids_cur_level+start_idx)
                start_idx+=distance_per_level.size(0)

            candidate_ids=torch.cat(candidate_ids,dim=0)
            overlaps = utils.compute_iou_xyxy(valid_anchors_per_img, gt_bbox)
            candidate_overlaps=overlaps[candidate_ids,range(num_gt)]
            overlap_mean=candidate_overlaps.mean(0)
            overlap_std=candidate_overlaps.std(0)
            overlap_threshold=overlap_mean+overlap_std

            is_pos = candidate_overlaps >= overlap_threshold[None,:]

            candidate_cx = (anchors_cx[candidate_ids.view(-1)]).reshape(-1,num_gt)
            candidate_cy = (anchors_cy[candidate_ids.view(-1)]).reshape(-1,num_gt)
            l = candidate_cx - gt_bbox[:, 0][None].contiguous()
            t = candidate_cy - gt_bbox[:, 1][None].contiguous()
            r = gt_bbox[:, 2][None].contiguous() - candidate_cx
            b = gt_bbox[:, 3][None].contiguous() - candidate_cy

            distance = torch.stack([l, t, r, b],dim=-1)
            is_inside = distance.min(dim=-1)[0] > 0.01

            is_pos = is_pos & is_inside

            for k in range(num_gt):
                candidate_ids[:, k] = candidate_ids[:, k]+k*num_valid_per_img
            overlaps_inf = torch.full_like(overlaps, -INF).t().contiguous().view(-1)
            index = candidate_ids.view(-1)[is_pos.view(-1)]
            overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
            overlaps_inf = overlaps_inf.view(num_gt, -1).t().contiguous()

            max_overlaps, argmax_overlaps=overlaps_inf.max(dim=1)
            assigned_gt_inds=overlaps.new_zeros((num_valid_per_img,),dtype=torch.long)
            assigned_gt_inds[max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF]+1

            pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(1)
            neg_inds = torch.nonzero(assigned_gt_inds == 0, as_tuple=False).squeeze(1)
            assigned_gt_inds=assigned_gt_inds[pos_inds]-1
            target_per_img=self.collect_targets(pos_inds,neg_inds,assigned_gt_inds,gt_bbox,gt_label,anchors,valid_per_img,strides)

            targets.append(target_per_img)

        return targets

    def collect_preds(self,cls_preds, reg_preds, targets):
        qfl_preds_batch=[]
        dfl_preds_batch=[]
        giou_preds_batch=[]
        qfl_targets_batch=[]
        dfl_targets_batch=[]
        giou_targets_batch=[]
        weight_batch=[]
        num_pos=0
        for i in range(len(targets)):
            target_per_img=targets[i]
            valid_per_img=target_per_img["valid"]
            pos_inds_per_img=target_per_img["pos_inds"]
            if(pos_inds_per_img.numel()==0):
                continue
            neg_inds_per_img=target_per_img["neg_inds"]
            pos_cls_labels_per_img=target_per_img["pos_cls_labels"]
            pos_reg_targets_per_img=target_per_img["pos_reg_targets"]
            giou_targets_batch.append(pos_reg_targets_per_img)
            pos_ltrb_targets_per_img=pos_reg_targets_per_img.clamp_(min=0,max=cfg.reg_max-0.1)
            dfl_targets_batch.append(pos_ltrb_targets_per_img)

            channels = reg_preds[0].size(1)
            reg_preds_per_img = [reg_pred[i].permute(1, 2, 0).view(-1, channels) for reg_pred in reg_preds]
            reg_preds_per_img = torch.cat(reg_preds_per_img, dim=0)
            reg_preds_per_img = reg_preds_per_img[valid_per_img, :]
            pos_reg_preds_per_img = reg_preds_per_img[pos_inds_per_img, :]
            dfl_preds_batch.append(pos_reg_preds_per_img)

            pos_ltrb_preds_per_img=self.intergral(pos_reg_preds_per_img)
            giou_preds_batch.append(pos_ltrb_preds_per_img)

            cls_preds_per_img=[cls_pred[i].permute(1,2,0).view(-1,self.num_classes) for cls_pred in cls_preds]
            cls_preds_per_img=torch.cat(cls_preds_per_img,dim=0)
            cls_preds_per_img=cls_preds_per_img[valid_per_img,:]
            # cls_preds_per_img=cls_preds_per_img.sigmoid()

            pos_cls_preds_per_img=cls_preds_per_img[pos_inds_per_img,:]
            neg_cls_preds_per_img=cls_preds_per_img[neg_inds_per_img,:]
            qfl_preds_batch.append(torch.cat([pos_cls_preds_per_img,neg_cls_preds_per_img],dim=0))

            pos_iou=utils.compute_iou_ltrb(pos_ltrb_preds_per_img.detach(),pos_reg_targets_per_img)

            qfl_score_per_img=pos_iou.new_zeros(size=(qfl_preds_batch[-1].size(0),cfg.num_classes))
            qfl_score_per_img[torch.arange(pos_inds_per_img.size(0)),pos_cls_labels_per_img]=pos_iou

            qfl_targets_batch.append(qfl_score_per_img)

            weight_per_img=pos_cls_preds_per_img.detach().sigmoid()
            weight_per_img=weight_per_img.max(1)[0]
            weight_batch.append(weight_per_img)
            num_pos+=pos_inds_per_img.size(0)
        if(num_pos==0):
            return [None,]*8
        qfl_preds_batch = torch.cat(qfl_preds_batch,dim=0)
        dfl_preds_batch = torch.cat(dfl_preds_batch,dim=0)
        giou_preds_batch = torch.cat(giou_preds_batch,dim=0)
        qfl_targets_batch = torch.cat(qfl_targets_batch,dim=0)
        dfl_targets_batch = torch.cat(dfl_targets_batch,dim=0)
        giou_targets_batch = torch.cat(giou_targets_batch,dim=0)
        weight_batch = torch.cat(weight_batch,dim=0)
        return qfl_preds_batch,dfl_preds_batch,giou_preds_batch,qfl_targets_batch,dfl_targets_batch,giou_targets_batch,weight_batch,num_pos

    def compute_strides(self,scales,device,dtype):
        strides=[]
        for i in range(len(scales)):
            scale=scales[i]
            value=cfg.fpn_strides[i]
            stride_per_level=torch.full(scale,value,device=device,dtype=dtype)
            stride_per_level=stride_per_level.reshape(-1).contiguous()
            strides.append(stride_per_level)
        return strides

    def forward(self,cls_preds, reg_preds, gt_bboxes, gt_labels, pad_img_shape):
        scales = [cls_pred.shape[-2:] for cls_pred in cls_preds]
        device=cls_preds[0].device
        dtype=cls_preds[0].dtype
        anchors=utils.compute_anchors(self.base_anchors,scales,device,dtype)
        anchors=torch.cat(anchors,dim=0)
        strides=self.compute_strides(scales,device,dtype)
        strides=torch.cat(strides,dim=0)

        valids=[]
        for i in range(len(gt_bboxes)):
            valids.append(utils.compute_valid_flag(pad_img_shape[i,:],scales,device))

        targets = self.compute_targets(anchors, valids, gt_bboxes, gt_labels, strides)

        (qfl_preds_batch,dfl_preds_batch,giou_preds_batch,qfl_targets_batch,dfl_targets_batch,giou_targets_batch,weight_batch,num_pos) \
            = self.collect_preds(cls_preds,reg_preds,targets)

        if(qfl_preds_batch is None):
            qfl_loss = torch.tensor(0, dtype=torch.float)
            dfl_loss = torch.tensor(0, dtype=torch.float)
            giou_loss = torch.tensor(0, dtype=torch.float)
            return {"qfl_loss": qfl_loss, "dfl_loss": dfl_loss, "giou_loss": giou_loss}
        num_pos=torch.tensor(num_pos,device=device)
        num_pos=utils.reduce_mean(num_pos).item()
        num_pos = max(num_pos, 1.0)
        num_weight = weight_batch.sum()
        num_weight=utils.reduce_mean(num_weight).item()

        dfl_preds_batch=dfl_preds_batch.view(-1,self.reg_max+1)
        dfl_targets_batch=dfl_targets_batch.view(-1,)

        qfl_loss=self.qfl_loss_func(qfl_preds_batch,qfl_targets_batch,avg_factor=num_pos)
        dfl_loss=self.dfl_loss_func(dfl_preds_batch,dfl_targets_batch,avg_factor=num_weight,weights=weight_batch[:,None].repeat(1,4).view(-1))
        giou_loss=self.giou_loss_func(giou_preds_batch,giou_targets_batch,avg_factor=num_weight,weights=weight_batch)

        return {"qfl_loss":qfl_loss,"dfl_loss":dfl_loss,"giou_loss":giou_loss}

if __name__ == "__main__":
    tt=open("a.txt",mode="a")
    tt.write("cccccc \n")

    for i in range(10):
        tt.write("ss \n")