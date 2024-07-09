#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
sys.path.append("/home/rothmeier/Documents/projects/RandomAugmentation/") # bad code

import argparse
import os
from itertools import chain, cycle
import cv2
import tqdm
import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.modeling import build_model
from detectron2.structures import Instances
from randaug.data import datasets
from randaug.engine.rand_trainer import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from detectron2.engine import DefaultPredictor
from randaug.models.detr import add_detr_config


from PIL import Image


def setup_frcnn(args, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file("configs/simulation/fasterrcnn.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'fasterrcnn'
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    return cfg

def setup_detr(args, threshold=0.5):
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.merge_from_file("configs/simulation/detr.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'detr'
    return cfg

def setup_retinanet(args, threshold=0.5):
    cfg = get_cfg()
    cfg.merge_from_file("configs/simulation/retinanet.yaml")
    cfg.eval_output = "./evaluation"
    cfg.network = 'retinanet'
    #cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold 
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--concat", action="store_true", help="concat images")
    parser.add_argument("--data", default="./", metavar="FILE", help="path to original data. Only needed when concatenating real and augmented data")

    return parser.parse_args(in_args)

def output(vis, fname):
    if args.show:
        print(fname)
        cv2.imshow("window", vis.get_image()[:, :, ::-1])
        cv2.waitKey()
    else:
        filepath = os.path.join(dirname, fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)

def concat_v(img1, img2):
    dst = Image.new('RGB', (img1.width, img1.height + img2.height))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (0, img1.height))
    return dst

def concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def filter_indices(outputs):
    instances = outputs["instances"]
    score_threshold = 0.5

    filter_mask = instances.scores > score_threshold
    indices = torch.nonzero(filter_mask).flatten().tolist()
    filtered_instances = Instances(
        image_size=instances.image_size,
        pred_classes=instances.pred_classes[indices],
        scores=instances.scores[indices],
        pred_boxes=instances.pred_boxes[indices]
    )
    return filtered_instances

def load_model(cfg):
    model = DefaultPredictor(cfg)
    return model

def visualize(im, out, dst):
    v = Visualizer(im,
                   metadata=MetadataCatalog.get("coco_2014_train"),
                   scale=0.5
            )
    out = v.draw_instance_predictions(out.to("cpu"))
    out.save(dst)
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    images = os.listdir(args.data)

    setup_funcs = [setup_frcnn, setup_detr, setup_retinanet]

    for setup in setup_funcs:
        cfg = setup(args)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        model = load_model(cfg)

        for image in images:
            im = os.path.join(args.data, image)
            im = np.array(Image.open(im))
            out = filter_indices(model(im))  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
            visualize(im, out, f"{args.data}/{cfg.network}_{image}")
