from .adverse import register_dataset as register_adverse_dataset
from detectron2.config import get_cfg


DEFAULT_DATASETS_ROOT = "../../datasets/CVPR24/datasets"

register_adverse_dataset(DEFAULT_DATASETS_ROOT)