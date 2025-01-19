#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
sys.path.append("/home/rothmeier/Documents/projects/RandomAugmentation/") # bad code

import argparse
import os
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from PIL import Image
from tqdm import tqdm
from randaug.data.datasets.yolo import load_annotation


def setup_model(args):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d0','--dir0', type=str, default='./imgs/reference')
    parser.add_argument('-d1','--dir1', type=str, default='./imgs/augmented')
    parser.add_argument('-aug', '--augmentation', type=str)
    return parser.parse_args(in_args)
    

class BBoxEvaluator():

    def __init__(self, threshold) -> None:
        self.keep_s, self.keep_m, self.keep_l = 0, 0, 0
        self.add_s, self.add_m, self.add_l = 0, 0, 0
        self.remove_s, self.remove_m, self.remove_l = 0, 0, 0
        self.threshold = threshold

    def evaluate(self, augmentation, custom):
        total_s = self.keep_s + self.add_s + self.remove_s
        total_m = self.keep_m + self.add_m + self.remove_m
        total_l = self.keep_l + self.add_l + self.remove_l

        results = [(self.keep_s / total_s),
                   (self.add_s / total_s),
                   (self.remove_s / total_s),
                   (self.keep_m / total_m),
                   (self.add_m / total_m),
                   (self.remove_m / total_m),
                   (self.keep_l / total_l),
                   (self.add_l / total_l),
                   (self.remove_l / total_l)]
        results = [f'{x:.3f}' for x in results]
        results.insert(0, augmentation)
        results.insert(1, custom)
        results = ','.join(results) # type: ignore
        
        with open("pami.txt", "a") as f:
            f.writelines(results+'\n')
        
        f.close()


    def compare_to_label(self, output, label):
        boxes_output = get_bboxes(output)
        boxes_gt = get_gt_boxes(label)
        self.compare_keep(boxes_gt, boxes_output)
        self.compare_delete(boxes_gt, boxes_output)
        self.compare_add(boxes_output, boxes_gt)

    def compare_to_image(self, original, augmented):
        boxes_original = get_bboxes(original)
        boxes_augmented = get_bboxes(augmented)
        self.compare_keep(boxes_original, boxes_augmented)
        self.compare_delete(boxes_original, boxes_augmented)
        self.compare_add(boxes_augmented, boxes_original)

    def compare_keep(self, boxes1, boxes2):
        for box1 in boxes1:
            for box2 in boxes2:
                iou = bb_intersection_over_union(box1, box2)
                self.increase_keep(iou, box1)
    
    def compare_delete(self, boxes1, boxes2):
        for box1 in boxes1:
            ious = [bb_intersection_over_union(box1, box2) > self.threshold for box2 in boxes2]
            if True not in ious:
                self.increase_delete(box1)

    def compare_add(self, boxes1, boxes2):
        for box1 in boxes1:
            ious = [bb_intersection_over_union(box1, box2) > self.threshold for box2 in boxes2]
            if True not in ious:
                self.increase_add(box1)

    def get_bb_area(self, box):
        width = box[2] - box[0] 
        height = box[3] - box[1]
        return width * height 
        
    def increase_keep(self, iou, box):
        area = self.get_bb_area(box)

        if area < 32**2:
            if iou > self.threshold:
                self.keep_s += 1
        elif area >= 32**2 and area < 96**2:
            if iou > self.threshold:
                self.keep_m += 1
        else:
            if iou > self.threshold:
                self.keep_l += 1

    def increase_add(self, box):
        area = self.get_bb_area(box)

        if area < 32**2:
            self.add_s += 1
        elif area >= 32**2 and area < 96**2:
            self.add_m += 1
        else:
            self.add_l += 1

    def increase_delete(self, box):
        area = self.get_bb_area(box)

        if area < 32**2:
            self.remove_s += 1
        elif area >= 32**2 and area < 96**2:
            self.remove_m += 1
        else:
            self.remove_l += 1


def get_bboxes(data):
    data = data['instances'].to('cpu')
    pred_classes =  data.pred_classes.tolist()
    pred_boxes = [box.tolist() for box in data.pred_boxes]
    vehicle_boxes = []
    for i, detection in enumerate(pred_classes):
        if detection == 2:
            vehicle_boxes.append(pred_boxes[i])
    return vehicle_boxes

def get_gt_boxes(data):
    bboxes = [annotation.bbox for annotation in data.annotations]
    return bboxes

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     args = parse_args()
#     logger = setup_logger()
#     logger.info("Arguments: " + str(args))
#     model, cfg = setup_model(args)

#     metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
#     reference_images = os.listdir(os.path.join(args.dir0, 'images'))
#     reference_labels = os.listdir(os.path.join(args.dir0, 'labels'))
#     augmented_images = os.listdir(args.dir1)
#     ending = os.path.splitext(os.listdir(args.dir1)[0])[-1].lower()  

#     bb_eval_label_orig = BBoxEvaluator(threshold=0.5)
#     bb_eval_label_aug = BBoxEvaluator(threshold=0.5)
#     bb_eval_orig_aug = BBoxEvaluator(threshold=0.5)

#     for file in tqdm(reference_images):
#         path0 = os.path.join(args.dir0, 'images', file)
#         path1 = os.path.join(args.dir1, file)
#         file_path = os.path.join(args.dir0, 'images', file)
#         label_path = os.path.join(args.dir0, 'labels', file)[:-4] + '.txt'
#         label = load_annotation(file_path, label_path)
        
#         if os.path.exists(path0):
#             path1 = path1[:-4] + ending
        
#             image1 = cv2.imread(path0)
#             image2 = cv2.imread(path1)

#             original = model(image1)
#             augmented = model(image2)

#             #bb_eval_label_orig.compare_to_label(original, label)
#             bb_eval_label_aug.compare_to_label(augmented, label)
#             bb_eval_orig_aug.compare_to_image(original, augmented)


#     #bb_eval_label_orig.evaluate(args.augmentation, 'gt_original')
#     bb_eval_label_aug.evaluate(args.augmentation, 'gt_augmented')
#     bb_eval_orig_aug.evaluate(args.augmentation, 'orignal_augmented')


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    model, cfg = setup_model(args)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    reference_images = os.listdir(os.path.join(args.dir0, 'images'))
    reference_labels = os.listdir(os.path.join(args.dir0, 'labels'))
    augmented_images = os.listdir(args.dir1)

    bb_eval_label_orig = BBoxEvaluator(threshold=0.5)
    bb_eval_label_aug = BBoxEvaluator(threshold=0.5)
    bb_eval_orig_aug = BBoxEvaluator(threshold=0.5)

    for file in tqdm(augmented_images):
        augmented_path = os.path.join(args.dir1, file)
        file_path = os.path.join(args.dir0, 'images', file)[:-4] + '.jpg'
        label_path = os.path.join(args.dir0, 'labels', file)[:-4] + '.txt'
        label = load_annotation(file_path, label_path)
        
        if os.path.exists(file_path):        
            image1 = cv2.imread(file_path)
            image2 = cv2.imread(augmented_path)

            original = model(image1)
            augmented = model(image2)

            #bb_eval_label_orig.compare_to_label(original, label)
            #bb_eval_label_aug.compare_to_label(augmented, label)
            bb_eval_orig_aug.compare_to_image(original, augmented)


    #bb_eval_label_orig.evaluate(args.augmentation, 'gt_original')
    #bb_eval_label_aug.evaluate(args.augmentation, 'gt_augmented')
    bb_eval_orig_aug.evaluate(args.augmentation, 'orignal_augmented')
