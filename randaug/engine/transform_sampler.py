import detectron2.data.transforms as T
import numpy as np
import sys

from detectron2.data.transforms.augmentation_impl import *
from randaug.data.transforms.transforms import *

from itertools import product


gan_transforms = [ 
    "CycleGANFogAugmentation",
    "CycleGANRainAugmentation",
    "CycleGANSnowAugmentation",
    "CUTFogAugmentation",
    "CUTRainAugmentation",
    "CUTSnowAugmentation",
    "StableDiffusionFogAugmentation",
    "StableDiffusionRainAugmentation",
    "StableDiffusionSnowAugmentation",
    "CycleDiffusionFogAugmentation",
    "CycleDiffusionRainAugmentation",
    "CycleDiffusionSnowAugmentation"
]

image_transforms = [
    "ColorAugmentation",
    "ShearXAugmentation",
    "ShearYAugmentation",
    "FogAugmentation",
    "SnowAugmentation",
    #"DropAugmentation",
    "RainAugmentation",
    "TranslateXAugmentation",
    "TranslateYAugmentation",
    "RotationAugmentation",
    "AutoContrastAugmentation",
    "InvertAugmentation",
    "EqualizeAugmentation",
    "SolarizeAugmentation",
    "PosterizeAugmentation",
    "ContrastAugmentation",
    "BrightnessAugmentation",
    "SharpnessAugmentation",
    "CutoutAugmentation"
]


class TransformSampler():

    def __init__(self, cfg):
        self.cfg = cfg
        np.random.seed(42) # random seed is required for parallelisation to have same magnitudes on every process

    def _sample_transforms(self, N):
        return [t for t in product(gan_transforms + image_transforms, repeat = N)]
        
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
        
    def _filter_image_augmentations(self, ops):
        filtered = self._filter_gan_augmentations(ops)
        filtered = self._filter_duplicates(filtered)
        return filtered
    
    def _filter_gan_augmentations(self, ops):
        if len(ops[0]) <= 1:
            return ops
        filtered = []
        for op in ops:
            second_aug = op[1][0]
            if second_aug not in gan_transforms:
                filtered.append(op)
        return filtered

    def _filter_duplicates(self, ops):
        if len(ops[0]) <= 1:
            return ops
        filtered = []
        for op in ops:
            if op[0][0] != op[1][0]:
                filtered.append(op)
        return filtered

    def grid_search(self):
        ops = self._sample_augmentation(magnitude=self.cfg.rand_M)
        ops = self._filter_image_augmentations(ops)
        print(len(ops))
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg, aug[1], aug[0]) for aug in augs]
        return augs
    
    def random_search(self):
        ops = self._sample_augmentation(magnitude=None)
        ops = self._filter_image_augmentations(ops)
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg, aug[1], aug[0]) for aug in augs]
        return augs
    
    def no_augmentation(self):
        return [RandomAugmentation(self.cfg, 1, [])]
    
    def test(self):
        return [RandomAugmentation(self.cfg, 1, [CutoutAugmentation(magnitude=4)])]
