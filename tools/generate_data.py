#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
sys.path.append("/home/rothmeier/Documents/projects/RandomAugmentation/") # bad code

import argparse
import os
from itertools import chain
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from randaug.data import datasets
from randaug.engine.rand_trainer import RandTrainer
from randaug.engine.transform_sampler import TransformSampler


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 5 # magnitude of transforms
    cfg.box_postprocessing = True # needs to be True
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--sampling_rate", type=int, default=10, help="Number of images to sample per augmentation policy")
    parser.add_argument("--magnitude", type=int, default=0, help="Magnitude of the transforms to be applied")
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        filepath = os.path.join(dirname, fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)            

    scale = 1.0
    sampler = TransformSampler(cfg, epochs=0)
    transforms = sampler.sample_output(args.magnitude)
    count = 0

    # images are saved in bounding box transform pipeline
    for t in transforms:
        train_data_loader = RandTrainer.build_train_loader(cfg=cfg, transforms=t.get_transforms())

        for batch in train_data_loader:
            count = count + 1
            if count % args.sampling_rate == 0:
                count = 0
                break
            else:
                continue


