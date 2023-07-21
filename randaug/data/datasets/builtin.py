from .midjourney_fog import register_dataset as register_midjourney_fog
from .midjourney_rain import register_dataset as register_midjourney_rain
from .midjourney_snow import register_dataset as register_midjourney_snow
from detectron2.config import get_cfg


DEFAULT_DATASETS_ROOT = "../../datasets/CVPR24/datasets"

register_midjourney_fog(DEFAULT_DATASETS_ROOT)
register_midjourney_rain(DEFAULT_DATASETS_ROOT)
register_midjourney_snow(DEFAULT_DATASETS_ROOT)