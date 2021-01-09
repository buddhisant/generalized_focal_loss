import torch
import torchvision
import cv2
import os

import config as cfg
import numpy as np

from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self,is_train=True,transforms=None):
        """
        初始化coco数据集
        :param ann_path: annotations文件的路径
        :param img_path: image文件夹的路径
        :param is_train: 当前是否是训练过程
        :param transforms: 数据预处理
        """
        if is_train:
            self.ann_path = cfg.train_ann_path
            self.img_path = cfg.train_img_path
        else:
            self.ann_path = cfg.val_ann_path
            self.img_path = cfg.val_img_path
        self.is_train = is_train
        self.transforms = transforms

        self.coco = COCO(self.ann_path)
        self.img_ids = self.get_imgIds()
        self.cat_ids = self.coco.getCatIds()

        # self.label2catid是连续标签到不连续标签的映射
        self.label2catid = {key:value for key,value in enumerate(self.cat_ids)}
        # self.catid2label是不连续标签到连续标签的映射
        self.catid2label = {value:key for key,value in enumerate(self.cat_ids)}

        # 获取数据集中图片的信息
        self.img_infos = self.get_imgInfos()

        self.img_flags = self.get_imgFlags()


    def get_imgFlags(self):
        """
        根据长宽比，对图片设置分组
        :return:
        """
        flags = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                flags[i] = 1
        return flags

    def __len__(self):
        """获取数据集长度"""
        return len(self.img_ids)

    def __getitem__(self, idx):
        if(self.is_train):
            img_info = self.img_infos[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_info["id"], iscrowd=False)
            anns = self.coco.loadAnns(ann_ids)

            gt_bboxes=[]
            gt_labels=[]

            for ann in anns:
                x1, y1, w, h = ann["bbox"]
                inter_w = max(0, min(x1+w, img_info["width"]) - max(x1, 0))
                inter_h = max(0, min(y1+h, img_info["height"]) - max(y1, 0))
                if inter_w*inter_h == 0:
                    continue
                if ann["area"] <= 0 or w < 1 or h < 1:
                    continue
                bbox = [x1, y1, x1 + w, y1 + h]
                gt_bboxes.append(bbox)
                gt_labels.append(self.catid2label[ann["category_id"]])

            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)

            img_name = img_info["file_name"]
            img_path = os.path.join(self.img_path,img_name)
            img = cv2.imread(img_path)

            transform_meta = {}
            img, gt_bboxes, gt_labels, transform_meta = self.transforms(img, gt_bboxes, gt_labels,transform_meta)

            ori_img_shape = transform_meta["ori_img_shape"]
            res_img_shape = transform_meta["res_img_shape"]
            pad_img_shape = transform_meta["pad_img_shape"]
            return {"images": img, "bboxes": gt_bboxes, "labels": gt_labels,"ori_img_shape":ori_img_shape,"res_img_shape":res_img_shape, "pad_img_shape":pad_img_shape}

        else:
            img_info = self.img_infos[idx]
            img_name = img_info["file_name"]
            img_path = os.path.join(self.img_path, img_name)
            img = cv2.imread(img_path)

            # inference时，不需要bbox等标注
            transform_meta={}
            img, _, _, transform_meta = self.transforms(img,transform_meta=transform_meta)

            ori_img_shape = transform_meta["ori_img_shape"]
            res_img_shape = transform_meta["res_img_shape"]
            pad_img_shape = transform_meta["pad_img_shape"]
            return {"images": img, "ori_img_shape": ori_img_shape, "res_img_shape":res_img_shape, "pad_img_shape":pad_img_shape, "indexs": idx}

    def get_imgIds(self):
        """
        得到数据集中的图片id，对于test模式下，不需要过滤图片；对于train模式，需要过滤图片，保留含有物体的图片
        :return:
        """
        img_ids=self.coco.getImgIds()
        if(not self.is_train):
            return img_ids
        vailed_img_ids=[]
        for i in range(len(img_ids)):
            annids = self.coco.getAnnIds(img_ids[i],iscrowd=False)
            if(len(annids)==0):
                continue
            vailed_img_ids.append(img_ids[i])
        return vailed_img_ids

    def get_imgInfos(self):
        """
        获取图片的信息
        :return:
        """
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs(i)[0]
            img_infos.append(info)
        return img_infos
