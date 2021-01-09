import torch
import config as cfg
import utils

import torch.nn.functional as F

class Intergral(torch.nn.Module):
    def __init__(self):
        super(Intergral, self).__init__()
        self.reg_max=cfg.reg_max
        self.register_buffer("project",torch.linspace(0,self.reg_max,self.reg_max+1))

    def forward(self,x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x

class Inference():
    def __init__(self):
        self.base_anchors = torch.tensor(cfg.base_anchors,dtype=torch.float)
        self.intergral=Intergral()
        self.reg_max=cfg.reg_max

    def ltrb_to_xyxy(self,ltrb,center,res_img_shape):
        x1=center[:,0]-ltrb[:,0]
        y1=center[:,1]-ltrb[:,1]
        x2=center[:,0]+ltrb[:,2]
        y2=center[:,1]+ltrb[:,3]
        x1.clamp_(min=0,max=res_img_shape[1])
        y1.clamp_(min=0,max=res_img_shape[0])
        x2.clamp_(min=0, max=res_img_shape[1])
        y2.clamp_(min=0, max=res_img_shape[0])
        return torch.stack([x1,y1,x2,y2],dim=-1)

    def __call__(self, cls_preds,reg_preds,res_img_shape,pad_img_shape):
        scales = [cls_pred.shape[-2:] for cls_pred in cls_preds]
        device = cls_preds[0].device
        dtype = cls_preds[0].dtype

        anchors = utils.compute_anchors(self.base_anchors, scales, device, dtype)

        res_img_shape=res_img_shape.squeeze(0)

        cls_preds=[cls_pred.squeeze(0).permute(1,2,0).reshape(-1,cfg.num_classes) for cls_pred in cls_preds]

        reg_preds=[reg_pred.squeeze(0).permute(1,2,0).reshape(-1,4*(self.reg_max+1)) for reg_pred in reg_preds]

        candidate_bboxes=[]
        candidate_scores=[]
        for i in range(len(cls_preds)):
            anchors_per_level=anchors[i]
            cls_preds_per_level=cls_preds[i]
            reg_preds_per_level=reg_preds[i]

            cls_preds_per_level=cls_preds_per_level.sigmoid()
            max_scores, max_inds=cls_preds_per_level.max(dim=1)
            topk=min(cfg.num_pred_before_nms,cls_preds_per_level.size(0))
            _, topk_inds=torch.topk(max_scores,k=topk)

            candidate_anchors_per_level=anchors_per_level[topk_inds,:]
            candidate_factors_per_level=reg_preds_per_level[topk_inds,:]
            anchor_cx = (candidate_anchors_per_level[:, 0] + candidate_anchors_per_level[:, 2]) * 0.5
            anchor_cy = (candidate_anchors_per_level[:, 1] + candidate_anchors_per_level[:, 3]) * 0.5
            candidate_ltrb_per_level=self.intergral(candidate_factors_per_level)*cfg.fpn_strides[i]
            candidate_bboxes_per_level = self.ltrb_to_xyxy(candidate_ltrb_per_level, torch.stack([anchor_cx, anchor_cy], dim=-1),res_img_shape)

            candidate_scores.append(cls_preds_per_level[topk_inds,:])
            candidate_bboxes.append(candidate_bboxes_per_level)

        candidate_bboxes=torch.cat(candidate_bboxes,dim=0)
        candidate_scores=torch.cat(candidate_scores,dim=0)

        pos_mask=candidate_scores>=cfg.pos_threshold
        pos_location = torch.nonzero(pos_mask, as_tuple=False)
        pos_inds = pos_location[:, 0]
        pos_labels = pos_location[:, 1]

        scores=candidate_scores[pos_inds,pos_labels]
        bboxes=candidate_bboxes[pos_inds,:]
        scores, bboxes, labels = utils.mc_nms(scores, bboxes, pos_labels, cfg.nms_threshold)
        scores = scores[:cfg.max_dets]
        bboxes = bboxes[:cfg.max_dets]
        labels = labels[:cfg.max_dets]

        return scores,bboxes,labels
