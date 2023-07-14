from dataclasses import dataclass


@dataclass
class DatasetInfo:
    name: str
    images_root: str
    annotations_fpath: str