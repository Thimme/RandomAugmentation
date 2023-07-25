import detectron2.data.transforms as T
import numpy as np
import sys

from detectron2.data.transforms.augmentation_impl import *
from randaug.data.transforms.magnitude_mapper import map_magnitude_to_args
from randaug.data.transforms.transforms import RandomAugmentation


# transforms = [ 
#     "FixedSizeCrop",
#     "RandomBrightness",
#     "RandomContrast",
#     "RandomCrop",
#     "RandomExtent",
#     "RandomFlip",
#     "RandomSaturation",
#     "RandomLighting",
#     "RandomRotation",
#     "Resize",
#     "ResizeScale",
#     "ResizeShortestEdge",
#     "RandomCrop_CategoryAreaConstraint",
#     "RandomResize",
#     "MinIoURandomCrop",
# ]
transforms = [ 
    "FixedSizeCrop",
    "RandomBrightness",
    "RandomFlip",
]

def randaugment(N, M):
    """Generate a set of distortions.
    Args:
    N: Number of augmentation transformations to
    apply sequentially.
    M: Magnitude for all the transformations.
    """
    samples_ops = np.random.choice(transforms, N)
    return [(op, M) for op in samples_ops]


def get_transforms(ops):
    transforms = [getattr(sys.modules[__name__], op[0]) for op in ops]
    args = [map_magnitude_to_args(op[0], [op[1]]) for op in ops]
    transforms = [t(**arg) for t, arg in zip(transforms, args)]
    return transforms


def build_rand_train_aug(cfg):
    ops = randaugment(cfg.rand_N, cfg.rand_M)
    augs = get_transforms(ops)
    return augs


class TransformSampler():

    def __init__(self, cfg, transforms):
        self.cfg = cfg
        self.transforms = transforms

    def grid_search(self):
        augs = build_rand_train_aug(self.cfg)
        return [RandomAugmentation(self.cfg, augs)]
    
    def random_search(self):
        return []