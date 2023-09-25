import detectron2.data.transforms as T
import numpy as np
import sys

from detectron2.data.transforms.augmentation_impl import *
from randaug.data.transforms.transforms import *

from itertools import product

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


# ShearX/Y,
# TranslateX/Y, Rotate, AutoContrast, Invert, Equalize, Solarize, Posterize, Contrast, Color, Brightness, Sharpness,
# Cutout, Sample Pairing
# apply flip with 0.5 to every frame


transforms = [ 
    "ColorAugmentation",
    "CycleGANAugmentation",
    "FogAugmentation",
    "RainAugmentation",
    "RandomFlip",
]

class TransformSampler():

    def __init__(self, cfg):
        self.cfg = cfg
        np.random.seed(42) # random seed is required for parallelisation to have same magnitudes on every process

    def _sample_transforms(self, N):
        return [t for t in product(transforms, repeat = N)]
        
    def _map_to_transforms(self, ops):
        transforms = [getattr(sys.modules[__name__], op[0]) for op in ops]
        transforms = [t(op[1]) for t, op in zip(transforms, ops)]
        magnitudes = [op[1] for op in ops]
        return transforms, magnitudes
    
    def _sample_augmentation(self, magnitude=None):
        augs = []
        transforms = self._sample_transforms(self.cfg.rand_N)
        for transform in transforms:
            aug = [(t, self._magnitude(magnitude)) for t in transform]
            augs.append(aug)

        return augs
    
    def _magnitude(self, m=None):
        if m == None:
            return np.random.randint(1, 10)
        else:
            return m

    def grid_search(self):
        ops = self._sample_augmentation(magnitude=self.cfg.rand_M)
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg.rand_N, aug[1], aug[0]) for aug in augs]
        return augs
    
    def random_search(self):
        ops = self._sample_augmentation(magnitude=None)
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg.rand_N, aug[1], aug[0]) for aug in augs]
        return augs
    
    def no_augmentation(self):
        return [RandomAugmentation(self.cfg.rand_N, 1, [])]
    
    def test(self):
        return [RandomAugmentation(self.cfg.rand_N, 1, [SolarizeAugmentation(magnitude=10), BrightnessAugmentation(magnitude=10)])]
