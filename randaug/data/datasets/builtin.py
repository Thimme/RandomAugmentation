from .midjourney import register_dataset as register_midjourney
from .dense365 import register_dataset as register_dense365
from .adverse import register_dataset as register_adverse
from .dawn import register_dataset as register_dawn
from .acdc import register_dataset as register_acdc
from .cityscapes import register_dataset as register_cityscapes
from .nuscenes import register_dataset as register_nuscenes
from .providentia import register_dataset as register_providentia
from .waymo import register_dataset as register_waymo
from .bdd import register_dataset as register_bdd
from ..data_classes import DatasetInfo
from detectron2.config import get_cfg


DEFAULT_DATASETS_ROOT = "/mnt/ssd2/dataset/cvpr24/"


# -------- MIDJOURNEY -----------
mj_snow = DatasetInfo(name="midjourney_snow", 
                      images_root="midjourney/snow/images",
                      annotations_fpath="midjourney/snow/labels")

mj_rain = DatasetInfo(name="midjourney_rain", 
                           images_root="midjourney/rain/images",
                           annotations_fpath="midjourney/rain/labels")

mj_fog = DatasetInfo(name="midjourney_fog", 
                           images_root="midjourney/fog/images",
                           annotations_fpath="midjourney/fog/labels")

# -------- ADVERSE -----------
adverse_clear = DatasetInfo(name="adverse_clear", 
                            images_root="adverse/clear/images",
                            annotations_fpath="adverse/clear/labels")

adverse_fog = DatasetInfo(name="adverse_fog", 
                          images_root="adverse/fog/images",
                          annotations_fpath="adverse/fog/labels")

adverse_rain = DatasetInfo(name="adverse_rain", 
                            images_root="adverse/rain/images",
                            annotations_fpath="adverse/rain/labels")

adverse_snow = DatasetInfo(name="adverse_snow", 
                          images_root="adverse/snow/images",
                          annotations_fpath="adverse/snow/labels")

# -------- DENSE -----------
dense_clear_day = DatasetInfo(name="dense_clear_day",
                    images_root="dense365/split/clear_day/images",
                    annotations_fpath="dense365/split/clear_day/labels")

dense_clear_night = DatasetInfo(name="dense_clear_night",
                    images_root="dense365/split/clear_night/images",
                    annotations_fpath="dense365/split/clear_night/labels")

dense_light_fog_day = DatasetInfo(name="dense_light_fog_day",
                    images_root="dense365/split/light_fog_day/images",
                    annotations_fpath="dense365/split/light_fog_day/labels")

dense_light_fog_night = DatasetInfo(name="dense_light_fog_night",
                    images_root="dense365/split/light_fog_night/images",
                    annotations_fpath="dense365/split/light_fog_night/labels")

dense_fog_day = DatasetInfo(name="dense_fog_day",
                    images_root="dense365/split/dense_fog_day/images",
                    annotations_fpath="dense365/split/dense_fog_day/labels")

dense_fog_night = DatasetInfo(name="dense_fog_night",
                    images_root="dense365/split/dense_fog_night/images",
                    annotations_fpath="dense365/split/dense_fog_night/labels")

dense_rain = DatasetInfo(name="dense_rain",
                    images_root="dense365/split/rain/images",
                    annotations_fpath="dense365/split/rain/labels")

dense_snow_day = DatasetInfo(name="dense_snow_day",
                    images_root="dense365/split/snow_day/images",
                    annotations_fpath="dense365/split/snow_day/labels")

dense_snow_night = DatasetInfo(name="dense_snow_night",
                    images_root="dense365/split/snow_night/images",
                    annotations_fpath="dense365/split/snow_night/labels")

# -------- DAWN -----------
dawn_fog = DatasetInfo(name="dawn_fog",
                       images_root="dawn/fog/images",
                       annotations_fpath="dawn/fog/labels")

dawn_rain = DatasetInfo(name="dawn_rain",
                       images_root="dawn/rain/images",
                       annotations_fpath="dawn/rain/labels")

dawn_sand = DatasetInfo(name="dawn_sand",
                       images_root="dawn/sand/images",
                       annotations_fpath="dawn/sand/labels")

dawn_snow = DatasetInfo(name="dawn_snow",
                       images_root="dawn/snow/images",
                       annotations_fpath="dawn/snow/labels")

# -------- NUSCENES -----------
nuscenes_clear = DatasetInfo(name="nuscenes_clear",
                       images_root="nuscenes/clear/images",
                       annotations_fpath="nuscenes/clear/labels")

nuscenes_rain = DatasetInfo(name="nuscenes_rain",
                            images_root="nuscenes/rain/images",
                            annotations_fpath="nuscenes/rain/labels")

# -------- BDD -----------
bdd_clear = DatasetInfo(name="bdd_clear",
                       images_root="bdd100k/clear/images",
                       annotations_fpath="bdd100k/clear/labels")

bdd_rain = DatasetInfo(name="bdd_rain",
                       images_root="bdd100k/rain/images",
                       annotations_fpath="bdd100k/rain/labels")

bdd_snow = DatasetInfo(name="bdd_snow",
                       images_root="bdd100k/snow/images",
                       annotations_fpath="bdd100k/snow/labels")

# -------- WAYMO -----------
waymo_clear = DatasetInfo(name="waymo_clear",
                          images_root="waymo/images",
                          annotations_fpath="waymo/labels")

# -------- PROVIDENTIA -----------
providentia_clear = DatasetInfo(name="providentia_clear",
                       images_root="providentia/clear/images",
                       annotations_fpath="providentia/clear/labels")

providentia_fog = DatasetInfo(name="providentia_fog",
                       images_root="providentia/fog/images",
                       annotations_fpath="providentia/fog/labels")

providentia_snow = DatasetInfo(name="providentia_snow",
                       images_root="providentia/snow/images",
                       annotations_fpath="providentia/snow/labels")


# -------- CITYSCAPES -----------
cityscapes_clear = DatasetInfo(name="cityscapes_clear",
                       images_root="cityscapes/clear/images",
                       annotations_fpath="cityscapes/clear/labels")

cityscapes_fog01 = DatasetInfo(name="cityscapes_fog01",
                       images_root="cityscapes/fog01/images",
                       annotations_fpath="cityscapes/fog01/labels")

cityscapes_fog02 = DatasetInfo(name="cityscapes_fog02",
                       images_root="cityscapes/fog02/images",
                       annotations_fpath="cityscapes/fog02/labels")

cityscapes_fog03 = DatasetInfo(name="cityscapes_fog03",
                       images_root="cityscapes/fog03/images",
                       annotations_fpath="cityscapes/fog03/labels")

# -------- ACDC -----------
acdc_clear = DatasetInfo(name="acdc_clear",
                       images_root="acdc/clear/images",
                       annotations_fpath="acdc/clear/labels")

acdc_fog = DatasetInfo(name="acdc_fog",
                       images_root="acdc/fog/images",
                       annotations_fpath="acdc/fog/labels")

acdc_rain = DatasetInfo(name="acdc_rain",
                       images_root="acdc/rain/images",
                       annotations_fpath="acdc/rain/labels")

acdc_snow = DatasetInfo(name="acdc_snow",
                       images_root="acdc/snow/images",
                       annotations_fpath="acdc/snow/labels")


register_midjourney(DEFAULT_DATASETS_ROOT, mj_fog)
register_midjourney(DEFAULT_DATASETS_ROOT, mj_rain)
register_midjourney(DEFAULT_DATASETS_ROOT, mj_snow)

register_adverse(DEFAULT_DATASETS_ROOT, adverse_clear)
register_adverse(DEFAULT_DATASETS_ROOT, adverse_fog)
register_adverse(DEFAULT_DATASETS_ROOT, adverse_rain)
register_adverse(DEFAULT_DATASETS_ROOT, adverse_snow)

register_dense365(DEFAULT_DATASETS_ROOT, dense_clear_day)
register_dense365(DEFAULT_DATASETS_ROOT, dense_clear_night)
register_dense365(DEFAULT_DATASETS_ROOT, dense_light_fog_day)
register_dense365(DEFAULT_DATASETS_ROOT, dense_light_fog_night)
register_dense365(DEFAULT_DATASETS_ROOT, dense_fog_day)
register_dense365(DEFAULT_DATASETS_ROOT, dense_fog_night)
register_dense365(DEFAULT_DATASETS_ROOT, dense_rain)
register_dense365(DEFAULT_DATASETS_ROOT, dense_snow_day)
register_dense365(DEFAULT_DATASETS_ROOT, dense_snow_night)

register_dawn(DEFAULT_DATASETS_ROOT, dawn_fog)
register_dawn(DEFAULT_DATASETS_ROOT, dawn_rain)
register_dawn(DEFAULT_DATASETS_ROOT, dawn_sand)
register_dawn(DEFAULT_DATASETS_ROOT, dawn_snow)

register_nuscenes(DEFAULT_DATASETS_ROOT, nuscenes_clear)
register_nuscenes(DEFAULT_DATASETS_ROOT, nuscenes_rain)

register_bdd(DEFAULT_DATASETS_ROOT, bdd_clear)
register_bdd(DEFAULT_DATASETS_ROOT, bdd_rain)
register_bdd(DEFAULT_DATASETS_ROOT, bdd_snow)

register_cityscapes(DEFAULT_DATASETS_ROOT, cityscapes_clear)
register_cityscapes(DEFAULT_DATASETS_ROOT, cityscapes_fog01)
register_cityscapes(DEFAULT_DATASETS_ROOT, cityscapes_fog02)
register_cityscapes(DEFAULT_DATASETS_ROOT, cityscapes_fog03)

register_acdc(DEFAULT_DATASETS_ROOT, acdc_clear)
register_acdc(DEFAULT_DATASETS_ROOT, acdc_rain)
register_acdc(DEFAULT_DATASETS_ROOT, acdc_fog)
register_acdc(DEFAULT_DATASETS_ROOT, acdc_snow)

register_providentia(DEFAULT_DATASETS_ROOT, providentia_clear)
register_providentia(DEFAULT_DATASETS_ROOT, providentia_fog)
register_providentia(DEFAULT_DATASETS_ROOT, providentia_snow)

# register_waymo(DEFAULT_DATASETS_ROOT, waymo_clear)