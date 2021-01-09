import torch
import math
import config as cfg

from torch.utils.data import DataLoader

def batch_padding(all_images):
    """
    将所有的image整合成为一个tensor，要按照一个batch最大图片的尺寸进行填充，但同时要能够被cfg.res_stride整除
    :param all_images: 类型为list
    :return:
    """
    #需要找到一个batch中，最大的尺寸
    max_size = [max(s) for s in zip(*[img.shape for img in all_images])]

    #batch_shape就是网络input的tensor的shape
    batch_shape = (len(all_images), max_size[0],max_size[1],max_size[2])
    batch_images = all_images[0].new(*batch_shape).zero_()
    for img, b_img in zip(all_images, batch_images):
        b_img[:, :img.shape[1], :img.shape[2]].copy_(img)

    return batch_images

class Collate():
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        """
        自定义的collect函数，被dataloader调用，用来整合来自于dataset的数据，构建网络的input
        :param batch:
        :return:
        """
        all_images = [s["images"] for s in batch]
        ori_img_shape = [s["ori_img_shape"] for s in batch]
        ori_img_shape = torch.tensor(ori_img_shape)
        res_img_shape = [s["res_img_shape"] for s in batch]
        res_img_shape = torch.tensor(res_img_shape)
        pad_img_shape = [s["pad_img_shape"] for s in batch]
        pad_img_shape = torch.tensor(pad_img_shape)

        if self.is_train:
            all_bboxes = [s["bboxes"] for s in batch]
            all_labels = [s["labels"] for s in batch]

            # 对一个batch中的图片进行填充，构建tensor
            batch_images = batch_padding(all_images)

            # 需要保留每张图片填充之前的尺寸
            return {"images": batch_images, "bboxes": all_bboxes, "labels": all_labels,"ori_img_shape":ori_img_shape,"res_img_shape":res_img_shape, "pad_img_shape":pad_img_shape}
        else:
            all_indexs = [s["indexs"] for s in batch]

            batch_images = batch_padding(all_images)
            return {"images": batch_images, "ori_img_shape":ori_img_shape, "res_img_shape":res_img_shape, "pad_img_shape":pad_img_shape, "indexs":all_indexs}

def build_dataloader(dataset, sampler, is_train=True):
    if(is_train):
        batch_size=cfg.samples_per_gpu
        num_workers=cfg.num_workers_per_gpu
        collate=Collate(is_train=is_train)
        dataloader=DataLoader(dataset,batch_size=batch_size,sampler=sampler, num_workers=num_workers, collate_fn=collate)
        # dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    else:
        collate = Collate(is_train=is_train)
        dataloader=DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate, pin_memory=False)
    return dataloader

