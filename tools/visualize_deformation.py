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
    return parser.parse_args(in_args)
    

class BBoxVisualizer():

    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.boxes_keep = []
        self.boxes_add = []
        self.boxes_delete = []

    def compare_to_image(self, original, augmented):
        boxes_original = get_bboxes(original)
        boxes_augmented = get_bboxes(augmented)
        self.draw_boxes(boxes_original, boxes_augmented)
        return self.boxes_keep, self.boxes_add, self.boxes_delete
    
    def compare_to_label(self, label, augmented):
        boxes_augmented = get_bboxes(augmented)
        boxes_gt = get_gt_boxes(label)
        self.draw_boxes(boxes_gt, boxes_augmented)
        return self.boxes_keep, self.boxes_add, self.boxes_delete

    def draw_boxes(self, boxes_original, boxes_augmented):
        self.draw_keep(boxes_original, boxes_augmented)
        self.draw_delete(boxes_original, boxes_augmented)
        self.draw_add(boxes_augmented, boxes_original)

    def draw_keep(self, boxes1, boxes2):
        for box1 in boxes1:
            for box2 in boxes2:
                iou = bb_intersection_over_union(box1, box2)
                if iou > self.threshold:
                    self.boxes_keep.append(box1)
    
    def draw_delete(self, boxes1, boxes2):
        for box1 in boxes1:
            ious = [bb_intersection_over_union(box1, box2) > self.threshold for box2 in boxes2]
            if True not in ious:
                self.boxes_delete.append(box1)

    def draw_add(self, boxes1, boxes2):
        for box1 in boxes1:
            ious = [bb_intersection_over_union(box1, box2) > self.threshold for box2 in boxes2]
            if True not in ious:
                self.boxes_add.append(box1)


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

def draw_boxes(image, keep, add, delete, file, original_img, labels):
    # Define colors in BGR format (OpenCV default):
    color_keep = (0, 165, 255)  # orange
    color_add = (0, 255, 0)     # green
    color_delete = (0, 0, 255)  # red

    print(image.dtype)
    # Draw "keep" boxes in orange
    for box in keep:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_keep, thickness=1)

    # Draw "add" boxes in green
    for box in add:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_add, thickness=1)

    # Draw "delete" boxes in red
    for box in delete:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color_delete, thickness=1)

    for box in get_gt_boxes(labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color_keep, thickness=1)

    # Finally, save the result. 
    # Change args.save_dir to wherever you want to store the visualized images
    save_path = os.path.join("tools/visualize_deformation", file)
    print(f"Save file to {save_path}")
    im_c = cv2.hconcat([original_img, image])
    cv2.imwrite(save_path, im_c)

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

    for file in tqdm(augmented_images):
        augmented_path = os.path.join(args.dir1, file)
        file_path = os.path.join(args.dir0, 'images', file)[:-4] + '.jpg'
        label_path = os.path.join(args.dir0, 'labels', file)[:-4] + '.txt'
        label = load_annotation(file_path, label_path)

        if os.path.exists(file_path):        
            image1 = cv2.imread(file_path)
            image2 = cv2.imread(augmented_path)

            original =  model(image1)
            augmented = model(image2)

            keep, add, delete = BBoxVisualizer(threshold=0.5).compare_to_label(label, augmented)
            #keep, add, delete = BBoxVisualizer(threshold=0.5).compare_to_image(original, augmented)
            draw_boxes(image2, keep, add, delete, file, image1, label)
