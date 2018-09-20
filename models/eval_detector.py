"""
Training script 4 Detection
"""
import os,sys
sys.path.append("/home/chuckgu/Data/HOI")


from dataloaders.mscoco import CocoDetection, CocoDataLoader
from dataloaders.visual_genome import VGDataLoader, VG
from dataloaders.hico_det import HICO,HICODataLoader
from lib.object_detector import ObjectDetector
import numpy as np
from torch import optim
import torch
import pandas as pd
import time

from config import ModelConfig, FG_FRACTION, RPN_FG_FRACTION, IM_SCALE, BOX_SCALE
from torch.nn import functional as F
from lib.fpn.box_utils import bbox_loss
import torch.backends.cudnn as cudnn
from pycocotools.cocoeval import COCOeval
from lib.pytorch_misc import optimistic_restore, clip_grad_norm
from torch.optim.lr_scheduler import ReduceLROnPlateau




cudnn.benchmark = True
conf = ModelConfig()

if conf.coco:
    train, val = CocoDetection.splits()
    val.ids = val.ids[:conf.val_size]
    train.ids = train.ids
    train_loader, val_loader = CocoDataLoader.splits(train, val, batch_size=conf.batch_size,
                                                     num_workers=conf.num_workers,
                                                     num_gpus=conf.num_gpus)
else:
    train, val = HICO.splits(num_val_im=conf.val_size)
    train_loader, val_loader = HICODataLoader.splits(train, val, batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus)

detector = ObjectDetector(classes=train.ind_to_classes, num_gpus=conf.num_gpus,
                          mode='rpntrain' if not conf.use_proposals else 'proposals', use_resnet=conf.use_resnet)
detector.cuda()


start_epoch = -1
if conf.ckpt is not None:
    ckpt = torch.load(conf.ckpt)
    if optimistic_restore(detector, ckpt['state_dict']):
        start_epoch = ckpt['epoch']



def val_epoch():
    detector.eval()
    # all_boxes is a list of length number-of-classes.
    # Each list element is a list of length number-of-images.
    # Each of those list elements is either an empty list []
    # or a numpy array of detection.
    vr = []
    for val_b, batch in enumerate(val_loader):
        vr.append(val_batch(val_b, batch))
    vr = np.concatenate(vr, 0)
    if vr.shape[0] == 0:
        print("No detections anywhere")
        return 0.0

    val_coco = val.coco
    coco_dt = val_coco.loadRes(vr)
    coco_eval = COCOeval(val_coco, coco_dt, 'bbox')
    coco_eval.params.imgIds = val.ids if conf.coco else [x for x in range(len(val))]

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    mAp = coco_eval.stats[1]
    return mAp


def val_batch(batch_num, b):
    result = detector[b]
    if result is None:
        return np.zeros((0, 7))
    scores_np = result.obj_scores.data.cpu().numpy()
    cls_preds_np = result.obj_preds.data.cpu().numpy()
    boxes_np = result.boxes_assigned.data.cpu().numpy()
    im_inds_np = result.im_inds.data.cpu().numpy()
    im_scales = b.im_sizes.reshape((-1, 3))[:, 2]
    if conf.coco:
        boxes_np /= im_scales[im_inds_np][:, None]
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        cls_preds_np[:] = [val.ind_to_id[c_ind] for c_ind in cls_preds_np]
        im_inds_np[:] = [val.ids[im_ind + batch_num * conf.batch_size * conf.num_gpus]
                         for im_ind in im_inds_np]
    else:
        # boxes_np *= BOX_SCALE / IM_SCALE
        boxes_np /= im_scales[im_inds_np][:, None]
        boxes_np[:, 2:4] = boxes_np[:, 2:4] - boxes_np[:, 0:2] + 1
        im_inds_np += batch_num * conf.batch_size * conf.num_gpus

    return np.column_stack((im_inds_np, boxes_np, scores_np, cls_preds_np))


print("Evaluation starts now!")
val_epoch()
