import torch
import math
import config as cfg

from loss import GFLLoss
from inference import Inference
from resnet import resnet
from fpn import PyramidFeatures

class Scaler(torch.nn.Module):
    def __init__(self):
        super(Scaler, self).__init__()
        self.scale=torch.nn.Parameter(torch.tensor(cfg.scale_init_value,dtype=torch.float32))

    def forward(self, input):
        return input*self.scale

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.num_anchors=len(cfg.base_anchors)//5

        self.conv1 = torch.nn.Conv2d(cfg.fpn_channels,cfg.fpn_channels,kernel_size=3, padding=1, bias=False)
        self.gn1 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv2 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv3 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1, bias=False)
        self.gn3 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv4 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1, bias=False)
        self.gn4 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)

        self.gfl_cls = torch.nn.Conv2d(cfg.fpn_channels,self.num_anchors*cfg.num_classes,kernel_size=3,padding=1)

        for m in self.modules():
            if(isinstance(m, torch.nn.Conv2d)):
                torch.nn.init.normal_(m.weight,mean=0.0,std=0.01)
                if(hasattr(m,"bias") and m.bias is not None):
                    torch.nn.init.constant_(m.bias,0)
            elif(isinstance(m, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias,0)
        bias_value = -math.log((1-cfg.class_prior_prob)/cfg.class_prior_prob)
        torch.nn.init.constant_(self.gfl_cls.bias,bias_value)

    def forward(self, x):
        cls_preds=[]
        for cls_feat in x:
            cls_feat = self.conv1(cls_feat)
            cls_feat = self.gn1(cls_feat)
            cls_feat.relu_()

            cls_feat = self.conv2(cls_feat)
            cls_feat = self.gn2(cls_feat)
            cls_feat.relu_()

            cls_feat = self.conv3(cls_feat)
            cls_feat = self.gn3(cls_feat)
            cls_feat.relu_()

            cls_feat = self.conv4(cls_feat)
            cls_feat = self.gn4(cls_feat)
            cls_feat.relu_()

            cls_feat = self.gfl_cls(cls_feat)
            cls_preds.append(cls_feat)

        return cls_preds

class Regressier(torch.nn.Module):
    def __init__(self):
        super(Regressier, self).__init__()
        self.num_anchors = len(cfg.base_anchors) // 5
        self.reg_max=cfg.reg_max
        self.conv1 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv2 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv3 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1, bias=False)
        self.gn3 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv4 = torch.nn.Conv2d(cfg.fpn_channels, cfg.fpn_channels, kernel_size=3, padding=1, bias=False)
        self.gn4 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)

        self.gfl_reg = torch.nn.Conv2d(cfg.fpn_channels,self.num_anchors*(self.reg_max+1)*4,kernel_size=3,padding=1)
        self.scales = torch.nn.ModuleList([Scaler() for _ in range(5)])

        for m in self.modules():
            if(isinstance(m, torch.nn.Conv2d)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if (hasattr(m, "bias") and m.bias is not None):
                    torch.nn.init.constant_(m.bias, 0)
            elif (isinstance(m, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        reg_preds=[]

        for i,reg_feat in enumerate(x):
            reg_feat = self.conv1(reg_feat)
            reg_feat = self.gn1(reg_feat)
            reg_feat.relu_()

            reg_feat = self.conv2(reg_feat)
            reg_feat = self.gn2(reg_feat)
            reg_feat.relu_()

            reg_feat = self.conv3(reg_feat)
            reg_feat = self.gn3(reg_feat)
            reg_feat.relu_()

            reg_feat = self.conv4(reg_feat)
            reg_feat = self.gn4(reg_feat)
            reg_feat.relu_()

            reg_preds.append(self.scales[i](self.gfl_reg(reg_feat)))

        return reg_preds

class GFL(torch.nn.Module):
    def __init__(self,is_train=True):
        super(GFL, self).__init__()
        self.is_train=is_train

        self.resNet = resnet()
        self.fpn = PyramidFeatures()

        self.classifier=Classifier()
        self.regressier=Regressier()

        self.loss = GFLLoss()
        self.inference = Inference()

    def forward(self,images, gt_bboxes=None, gt_labels=None, ori_img_shapes=None, res_img_shapes=None, pad_img_shapes=None):
        c3,c4,c5=self.resNet(images)
        features=self.fpn([c3,c4,c5])

        cls_preds=self.classifier(features)
        reg_preds=self.regressier(features)

        if self.is_train:
            losses = self.loss(cls_preds, reg_preds, gt_bboxes, gt_labels, pad_img_shapes)
            return losses
        else:
            scores, bboxes, labels = self.inference(cls_preds, reg_preds, res_img_shapes, pad_img_shapes)
            scale_factor = ori_img_shapes.float() / res_img_shapes.float()
            scale_factor = torch.cat([scale_factor, scale_factor], dim=1)
            bboxes = bboxes * scale_factor
            return scores, bboxes, labels