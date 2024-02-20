from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from ..data_classes import DatasetInfo
from dataclasses import dataclass
import numpy as np
import os
import cv2


class DetectionData:
    def __init__(self, file_name: str):
        self.filename = file_name
        self.image_id = file_name.split('/')[-1][:-4]
        self.annotations = list()
        self.calculate_size()


    def calculate_size(self):
        im = cv2.imread(self.filename)
        self.width = im.shape[1]
        self.height = im.shape[0]
    
    
    def add_annotation(self, annotation):
        self.annotations.append(annotation)


    def to_dict(self):
        return {
            "file_name": self.filename, 
            "height": self.height,
            "width": self.width,
            "image_id": self.image_id, 
            "annotations": [ann.to_dict() for ann in self.annotations]
        }


@dataclass
class AnnotationData:
    bbox: list
    category_id: int

    def to_dict(self):
        return {
            "bbox": self.bbox, 
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": self.category_id
        }


def load_annotation(image: str, annotation_fpath: str):
    with open(annotation_fpath, "r") as f:
        labels = f.readlines()
        detection = DetectionData(file_name=image)

        for label in labels: 
            split = label.strip().split(' ')
            x, y, width, height = float(split[1]), float(split[2]), float(split[3]), float(split[4])
            xmin = (x - width / 2.0) * detection.width
            ymin = (y - height / 2.0) * detection.height
            xmax = (x + width / 2.0) * detection.width
            ymax = (y + height / 2.0) * detection.height
            
            bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
            category = 2 # int(split[0])
            annotation = AnnotationData(bbox=bbox,
                                        category_id=category)
            detection.add_annotation(annotation)

        return detection


def load_label_for_image(image: str, annotations_fpath: str):
    annotation = image.split('/')[-1][:-4] + ".txt"
    return os.path.join(annotations_fpath, annotation)


def load_yolo_annotations(images_root: str, annotations_fpath: str):
    images = list(sorted(os.listdir(images_root)))
    images = [os.path.join(images_root, image) for image in images]
    #annotations = list(sorted(os.listdir(annotations_fpath)))
    #annotations = [os.path.join(annotations_fpath, annotation) for annotation in annotations]

    yolo_annotations = []

    for img in images:
        annotation = load_label_for_image(img, annotations_fpath)
        if os.path.isfile(annotation):
            yolo_annotations.append(load_annotation(img, annotation))
    
    dict = [ann.to_dict() for ann in yolo_annotations]
    return dict


def load_yolo_classes():
    return load_coco_classes()
    # return ["background", "car"]

def load_coco_classes():
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
        'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
