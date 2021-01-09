import os
import gc
import time
import utils
import torch
import argparse
import sampler
import dataset
import transform
import config as cfg

from tqdm import tqdm
from gfl import GFL
from dataloader import build_dataloader

def test(epochs_tested):
    is_train=False
    transforms = transform.build_transforms(is_train=is_train)
    coco_dataset = dataset.COCODataset(is_train=is_train, transforms=transforms)
    dataloader = build_dataloader(coco_dataset, sampler=None, is_train=is_train)

    assert isinstance(epochs_tested, (list, set)), "during test, archive_name must be a list or set!"
    model = GFL(is_train=is_train)

    for epoch in epochs_tested:
        utils.load_model(model, epoch)
        model.cuda()
        model.eval()

        final_results = []

        with torch.no_grad():
            for data in tqdm(dataloader):
                img = data["images"]
                ori_img_shape = data["ori_img_shape"]
                res_img_shape = data["res_img_shape"]
                pad_img_shape = data["pad_img_shape"]
                index = data["indexs"]

                img = img.cuda()
                ori_img_shape = ori_img_shape.cuda()
                res_img_shape = res_img_shape.cuda()
                pad_img_shape = pad_img_shape.cuda()

                cls_pred, reg_pred, label_pred = model(img,ori_img_shapes=ori_img_shape, res_img_shapes=res_img_shape,pad_img_shapes=pad_img_shape)

                cls_pred = cls_pred.cpu()
                reg_pred = reg_pred.cpu()
                label_pred = label_pred.cpu()
                index = index[0]

                img_info = dataloader.dataset.img_infos[index]
                imgid = img_info["id"]

                reg_pred = utils.xyxy2xywh(reg_pred)

                label_pred = label_pred.tolist()
                cls_pred = cls_pred.tolist()

                final_results.extend(
                    [
                        {
                            "image_id": imgid,
                            "category_id": dataloader.dataset.label2catid[label_pred[k]],
                            "bbox": reg_pred[k].tolist(),
                            "score": cls_pred[k],
                        }
                        for k in range(len(reg_pred))
                    ]
                )


        output_path = os.path.join(cfg.output_path, cfg.check_prefix+"_"+str(epoch)+".json")
        utils.evaluate_coco(dataloader.dataset.coco, final_results, output_path, "bbox")

def main():
    epochs_tested=[5,]
    utils.mkdir(cfg.output_path)
    test(epochs_tested)

if __name__ == "__main__":
    main()
