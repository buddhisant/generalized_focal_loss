import cv2
import math
import random
import config as cfg
import numpy as np
import torch

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, bboxs=None, labels=None, transform_meta=None):
        for t in self.transforms:
            image, bboxs, labels, transform_meta = t(image, bboxs, labels, transform_meta)
        return image, bboxs, labels, transform_meta

class Resize(object):
    def __init__(self):
        self.max_length = max(cfg.resize_scale)
        self.min_length = min(cfg.resize_scale)

    def get_target_scale(self,ori_scale):
        """
        计算放缩的目标尺寸
        :param ori_scale: 图片原始尺寸，格式为(h,w)
        :return: 放缩图片的目标尺寸，格式为(h,w)
        """
        h, w=ori_scale
        ori_min=min(h,w)
        ori_max=max(h,w)

        scale_factor=min(self.max_length/ori_max, self.min_length/ori_min)
        fin_min=int(ori_min * scale_factor+0.5)
        fin_max=int(ori_max * scale_factor+0.5)

        fin_scale=(fin_max,fin_min) if h>w else (fin_min,fin_max)
        return fin_scale

    def __call__(self, image, bboxs=None, labels=None, transform_meta=None):
        """
        对图片进行resize的预处理
        :param image: 类型为np.ndarray
        :param bboxs: xyxy格式的gt bbox, 类型为np.ndarray
        :param labels: gt labels, 类型为np.ndarray
        :return:
        """
        h, w = image.shape[:2]

        #fin_h和fin_w表示图片放缩的目标尺寸
        fin_h, fin_w = self.get_target_scale((h,w))

        resized_img=cv2.resize(image,(fin_w,fin_h),interpolation=cv2.INTER_LINEAR)

        if bboxs is not None:
            # 还需要对gt bbox做相应的放缩，并且横向和纵向的放缩因子有细微的不同
            scale_factor_w = fin_w / w
            scale_factor_h = fin_h / h
            scale_factor = np.array([scale_factor_w, scale_factor_h, scale_factor_w, scale_factor_h], dtype=np.float32)

            bboxs = bboxs * scale_factor
            bboxs[:, 0::2] = np.clip(bboxs[:, 0::2], 0, fin_w)
            bboxs[:, 1::2] = np.clip(bboxs[:, 1::2], 0, fin_h)
        if(transform_meta is not None):
            transform_meta["ori_img_shape"]=image.shape[:2]
            transform_meta["res_img_shape"]=resized_img.shape[:2]

        return resized_img, bboxs, labels,transform_meta

class RandomHorizontalFlip(object):
    def __init__(self):
        self.prob=cfg.flip_pro

    def __call__(self, image, bboxs=None, labels=None,transform_meta=None):
        """
        对图片进行水平翻转
        :param image: 类型为np.ndarray
        :param bboxs: xyxy格式的gt bbox, 类型为np.ndarray
        :param labels: gt labels, 类型为np.ndarray
        :return:
        """
        if random.random() < self.prob:
            image = np.flip(image, axis=1)
            if bboxs is not None:
                h, w = image.shape[:2]
                x1 = bboxs[:, 0]
                y1 = bboxs[:, 1]
                x2 = bboxs[:, 2]
                y2 = bboxs[:, 3]
                new_x1 = w-x2
                new_x2 = w-x1
                new_bboxs = np.stack([new_x1, y1, new_x2, y2], axis=1)
                bboxs = new_bboxs
        return image, bboxs, labels, transform_meta

class Normalize(object):
    def __init__(self):
        #normalize操作的均值
        self.mean=np.array(cfg.norm_mean,dtype=np.float64).reshape(1,-1)
        #normalize操作的标准差
        self.std=np.array(cfg.norm_std,dtype=np.float64).reshape(1,-1)
        #将图片由bgr格式转化为rgb格式
        self.to_rgb=cfg.norm_to_rgb

    def __call__(self, image,bboxs=None, labels=None,transform_meta=None):
        image = image.astype(np.float32)
        if(self.to_rgb): #如果是采用caffe风格的预训练，则无需转化为RGB格式，如果是pytorch风格的预训练，需要转化为RGB
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=cv2.subtract(image, self.mean)
        stdinv = 1/self.std
        image=cv2.multiply(image, stdinv)
        return image, bboxs, labels, transform_meta

class Padding(object):
    def __init__(self):
        self.size_divisor=32

    def __call__(self, image, bboxs=None, labels=None,transform_meta=None):
        h, w=image.shape[:2]
        new_h=int(math.ceil(h/self.size_divisor))*self.size_divisor
        new_w=int(math.ceil(w/self.size_divisor))*self.size_divisor
        right_padding=new_w-w
        bottem_padding=new_h-h
        image=cv2.copyMakeBorder(image,0,bottem_padding,0,right_padding,cv2.BORDER_CONSTANT,value=0)
        if(transform_meta is not None):
            transform_meta["pad_img_shape"]=image.shape[:2]

        return image,bboxs,labels,transform_meta

class ToTensor(object):
    def __call__(self, image, bboxs=None, labels=None,transform_meta=None):
        image = image.transpose(2,0,1)
        image = torch.from_numpy(image).contiguous()

        if bboxs is not None:
            bboxs = torch.from_numpy(bboxs)
            labels = torch.from_numpy(labels)

        return image, bboxs, labels, transform_meta

def build_transforms(is_train=True):
    resize=Resize()
    randomHorizontalFlip=RandomHorizontalFlip()
    normalize=Normalize()
    padding=Padding()
    toTensor=ToTensor()
    if is_train:
        transforms=Compose([resize, randomHorizontalFlip, normalize, padding, toTensor])
    else:
        transforms=Compose([resize, normalize, padding, toTensor])
    return transforms