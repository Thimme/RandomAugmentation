from typing import Optional
from detectron2.data import DatasetCatalog, MetadataCatalog
from .yolo import load_yolo_annotations, load_yolo_classes
from ..data_classes import DatasetInfo
from ..utils import maybe_prepend_base_path

dataset_data = DatasetInfo(name="midjourney_rain", 
                           images_root="midjourney_rain/images",
                           annotations_fpath="midjourney_rain/labels")


def register_dataset(datasets_root: Optional[str] = None) -> None:
    
    annotations_fpath = maybe_prepend_base_path(datasets_root, dataset_data.annotations_fpath)
    images_root = maybe_prepend_base_path(datasets_root, dataset_data.images_root)

    def load_annotations():
        return load_yolo_annotations(
            images_root=images_root,
            annotations_fpath=annotations_fpath
        )
    
    DatasetCatalog.register(dataset_data.name, load_annotations)
    MetadataCatalog.get(dataset_data.name).set(
        image_root=dataset_data.images_root,
        annotations_fpath = dataset_data.annotations_fpath,
        thing_classes=load_yolo_classes()
    )
