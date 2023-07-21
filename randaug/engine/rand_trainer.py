import os
import detectron2.data.transforms as T
from randaug.data.transforms import weather_transforms as W
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetMapper
from detectron2.config import get_cfg


def build_rand_train_aug(cfg):
    augs = [
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        )
    ]
    if cfg.INPUT.CROP.ENABLED:
        augs.append(
            T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )
        )
    augs.append(T.RandomFlip())
    augs.append(W.MyColorAugmentation())
    augs.append(W.FogAugmentation(severity=5))
    return augs 


class RandTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_dir = os.path.join(cfg.eval_output, dataset_name)
        return COCOEvaluator(dataset_name, output_dir=output_dir)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_rand_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
