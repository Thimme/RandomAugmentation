#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
sys.path.append("/home/rothmeier/Documents/projects/RandomAugmentation/") # bad code

import argparse
import os
from itertools import chain
import cv2
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from randaug.data import datasets
from randaug.engine.rand_trainer import RandTrainer
from randaug.engine.transform_sampler import TransformSampler, RandomSampler


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.aug_prob = 1.0
    cfg.rand_N = args.augmentations
    cfg.rand_M = args.magnitude
    cfg.box_postprocessing = False
    cfg.save_image = True
    cfg.save_path = f'tools/deformation_improved/{cfg.rand_N}/{cfg.rand_M}'
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--samples", type=int, default=5000, help="Number of images to sample per augmentation policy")
    parser.add_argument("--augmentations", type=int, default=1, help="Number of sequential augmentations")
    parser.add_argument("--magnitude", type=int, default=0, help="Intensity of augmentation")

    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    os.makedirs(cfg.save_path, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        filepath = os.path.join(cfg.save_path, fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)   

    # images are saved in bounding box transform pipeline
    train_data_loader = RandTrainer.build_rand_augment_train_loader(cfg=cfg)
    count = 0
    pbar = tqdm(total=args.samples)

    for batch in train_data_loader: # type: ignore
        count += 1
        pbar.update(1)
        if count > args.samples:
            break
    
    pbar.close()

