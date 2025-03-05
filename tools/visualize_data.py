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

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from randaug.data import datasets
from randaug.engine.rand_trainer import RandTrainer
from randaug.engine.transform_sampler import TransformSampler
from PIL import Image


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.rand_N = 2 # number of transforms
    cfg.rand_M = 4 # magnitude of transforms
    #cfg.INPUT.FORMAT = "BGR"
    cfg.box_postprocessing = False
    cfg.cutout_postprocessing = False
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("--concat", action="store_true", help="concat images")
    parser.add_argument("--data", default="./", metavar="FILE", help="path to original data. Only needed when concatenating real and augmented data")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

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
        
    def visualize_image(img, instances) -> Visualizer:
        visualizer = Visualizer(img, metadata=metadata, scale=1.0)
        target_fields = instances.get_fields()
        labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
            masks=target_fields.get("gt_masks", None),
            keypoints=target_fields.get("gt_keypoints", None),
        )
        return vis
    
    def center_crop_and_resize(image_to_modify, target_image):
        target_width, target_height = target_image.size
        target_aspect = target_width / target_height
        orig_width, orig_height = image_to_modify.size
        orig_aspect = orig_width / orig_height
        
        # Center crop to the correct aspect ratio
        if orig_aspect > target_aspect:
            # If the original image is wider than the target, crop the left and right edges
            new_width = int(orig_height * target_aspect)
            left = (orig_width - new_width) / 2
            crop_rectangle = (left, 0, left + new_width, orig_height)
        else:
            # If the original image is taller than the target, crop the top and bottom edges
            new_height = int(orig_width / target_aspect)
            top = (orig_height - new_height) / 2
            crop_rectangle = (0, top, orig_width, top + new_height)
        
        cropped_image = image_to_modify.crop(crop_rectangle)
        resized_image = cropped_image.resize((target_width, target_height), Image.ANTIALIAS)
        return resized_image

    def sample(magnitude):
        if args.source == "dataloader":
            sampler = TransformSampler(cfg, epochs=0)
            transforms = sampler.test(magnitude=magnitude)
            train_data_loader = RandTrainer.build_train_loader(cfg=cfg, transforms=transforms[0].get_transforms())

            for batch in train_data_loader:
                for per_image in batch:
                    # Pytorch tensor is in (C, H, W) format
                    #print(per_image)
                    img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                    img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)
                    file_id = per_image['image_id']
                    img = Image.fromarray(img)
                    #img.save(os.path.join(dirname, file_id + '.jpg'))
                    #break
                    vis = visualize_image(img, per_image["instances"])

                    if args.concat:
                        # exchange with original bounding boxes
                        img2 = Image.fromarray(vis.get_image())
                        img1 = Image.open(os.path.join(args.data, file_id + '.jpg')) # original file path
                        img1 = center_crop_and_resize(img1, img2)
                        concat_image = concat_h(img1, img2)
                        concat_image.save(os.path.join(dirname, file_id + '.jpg'))
                    else:
                        output(vis, f"{magnitude}_{file_id}.jpg")
                    break
                break
        else:
            dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TRAIN]))
            if cfg.MODEL.KEYPOINT_ON:
                dicts = filter_images_with_few_keypoints(dicts, 1)
            for dic in tqdm.tqdm(dicts):
                img = utils.read_image(dic["file_name"], "RGB")
                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                vis = visualizer.draw_dataset_dict(dic)
                output(vis, os.path.basename(dic["file_name"]))

    for i in range(0,10):
        sample(magnitude=i)

