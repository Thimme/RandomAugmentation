import detectron2.data.transforms as T
import numpy as np
import sys
import random
from detectron2.data.transforms.augmentation_impl import *
from randaug.data.transforms.transforms import *

from itertools import product

gan_transforms = [
    #"CycleGANFogAugmentation",
    #"CycleGANRainAugmentation",
    #"CycleGANSnowAugmentation",
    #"CUTFogAugmentation",
    #"CUTRainAugmentation",
    #"CUTSnowAugmentation",
]

diffusion_transforms = [
    #"CycleDiffusionFogAugmentation",
    #"CycleDiffusionRainAugmentation",
    #"CycleDiffusionSnowAugmentation",
    "StableDiffusionFogAugmentation",
    "StableDiffusionRainAugmentation",
    "StableDiffusionSnowAugmentation",
    #"PlugPlayFogAugmentation",
    #"PlugPlayRainAugmentation",
    #"PlugPlaySnowAugmentation",
    #"ControlNetFogAugmentation",
    #"ControlNetRainAugmentation",
    #"ControlNetSnowAugmentation",
    #"MGIEDiffusionFogAugmentation",
    #"MGIEDiffusionRainAugmentation",
    #"MGIEDiffusionSnowAugmentation",
]


ai_transforms = [
    "CycleGAN",
    "CUT",
    "StableDiffusion",
    "CycleDiffusion",
    "MGIEDiffusion",
]

ai_conditions = [
    "FogAugmentation",
    "RainAugmentation",
    "SnowAugmentation",
]

image_transforms = [
    "ColorAugmentation",
    "ShearXAugmentation",
    "ShearYAugmentation",
    "FogAugmentation",
    "SnowAugmentation",
    "RainAugmentation",
    "TranslateXAugmentation",
    "TranslateYAugmentation",
    "AutoContrastAugmentation",
    "InvertAugmentation",
    "EqualizeAugmentation", #
    "SolarizeAugmentation",
    "PosterizeAugmentation",
    "ContrastAugmentation",
    "BrightnessAugmentation",
    "SharpnessAugmentation",
    "CutoutAugmentation",
    "DropAugmentation",
    "RotationAugmentation",
]

NUM_DATASETS = 32

class TransformSampler():

    def __init__(self, cfg, epochs):
        self.cfg = cfg
        self.epochs = self._get_epochs(epochs)
        np.random.seed(42) # random seed is required for parallelisation to have same magnitudes on every process

    def _sample_transforms(self, N):
        return [t for t in product(gan_transforms + diffusion_transforms + image_transforms, repeat = N)]
        
    def _map_to_transforms(self, ops):
        transforms = [getattr(sys.modules[__name__], op[0]) for op in ops]
        transforms = [t(op[1], cfg=self.cfg) for t, op in zip(transforms, ops)]
        magnitudes = [op[1] for op in ops]
        return transforms, magnitudes
    
    def _sample_augmentation(self, magnitude=None):
        augs = []
        transforms = self._sample_transforms(self.cfg.rand_N)
        for transform in transforms:
            aug = [(t, self._magnitude(magnitude)) for t in transform]
            augs.append(aug)

        return augs
    
    def _sample_diffusion_models(self, ids=[]):
        augs = []
        for transform in diffusion_transforms:
            _ = [augs.append([(transform, id)]) for id in ids]

        return augs
    
    def _magnitude(self, m=None):
        if m == None:
            return np.random.randint(1, 5)
        else:
            return m
        
    def _filter_image_augmentations(self, ops):
        filtered = self._filter_gan_augmentations(ops)
        filtered = self._filter_duplicates(filtered)
        #filtered = self._filter_non_weather(filtered)
        return filtered
    
    def _filter_gan_augmentations(self, ops):
        if len(ops[0]) <= 1:
            return ops
        filtered = []
        for op in ops:
            second_aug = op[1][0]
            if second_aug not in gan_transforms + diffusion_transforms:
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
    
    def _filter_non_weather(self, ops):
        weather = ["FogAugmentation", "SnowAugmentation", "RainAugmentation", "DropAugmentation"]
        if len(ops[0]) <= 1:
            return ops
        filtered = []
        for op in ops:
            if op[0][0] in weather or op[1][0] in weather:
                filtered.append(op)
        return filtered
    
    def _get_epochs(self, epochs):
        return int(epochs / NUM_DATASETS)

    def grid_search(self):
        ops = self._sample_augmentation(magnitude=self.cfg.rand_M)
        ops = self._filter_image_augmentations(ops)
        ops = ops[:57]
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg, aug[1], aug[0]) for aug in augs]
        print(f"Start from epoch: {self.epochs}")
        return augs[self.epochs:]
    
    def random_search(self):
        ops = self._sample_augmentation(magnitude=None)
        ops = self._filter_image_augmentations(ops)
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg, aug[1], aug[0]) for aug in augs]
        return augs

    def diffusion_search(self):
        ops = self._sample_diffusion_models(ids=[1])
        augs = [self._map_to_transforms(op) for op in ops] 
        augs = [RandomAugmentation(self.cfg, aug[1], aug[0]) for aug in augs]
        print(augs)
        return augs
    
    def no_augmentation(self):
        return [RandomAugmentation(self.cfg, 1, [])]
    
    def test(self):
        #return [RandomAugmentation(self.cfg, 1, [RotationAugmentation(magnitude=0), ColorAugmentation(magnitude=4)])]
        return [RandomAugmentation(self.cfg, 1, [StableDiffusionRainAugmentation(magnitude=1, cfg=self.cfg)])]
        #return [RandomAugmentation(self.cfg, 1, [])]
    
    def sample_output(self, magnitude=0):
        # amount of images
        # magnitudes (1-5 or only 5)
        # every gan + every image augmentation
        # how many samples per augmentation
        transforms = list(product(gan_transforms + diffusion_transforms, image_transforms))
        ops = []
        for transform in transforms:
            aug = [(t, self._magnitude(magnitude)) for t in transform]
            ops.append(aug)
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg, aug[1], aug[0]) for aug in augs]
        return augs


class RandomSampler():

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

    def _sample_ops(self, N):
        transforms = ai_transforms + image_transforms
        one_time = ai_transforms # ["DropAugmentation"]

        while True:
            # Sample elements from A
            sampled = np.random.choice(transforms, N) # type: ignore
            # Count how many elements are in B
            gan_count = sum(element in one_time for element in sampled)

            # It does not make sense to have more than one gan / diffusion transform as it replaces the rest of the augmentations
            if gan_count <= 1:
                sampled = [self._replace_sample(s) for s in sampled]
                return sampled
            
    def _replace_sample(self, samplloade):
        if sample in ai_transforms:
            condition = random.choice(ai_conditions)
            return f"{sample}{condition}"
        else:
            return sample
    
    def _add_magnitude(self, ops, M):
        magnitude = M
        return [(o, magnitude) for o in ops]
    
    def _map_to_transforms(self, ops):
        transforms = [getattr(sys.modules[__name__], op[0]) for op in ops]
        transforms = [self._init_transform(t, op[1]) for t, op in zip(transforms, ops)]
        magnitudes = [op[1] for op in ops]
        return transforms, magnitudes
    
    # Random sampling causes the drop augmentation not to distribute across devices so this function is needed
    def _init_transform(self, transform, magnitude) -> T.Augmentation:
        if transform == DropAugmentation:
            return transform(magnitude, cfg=self.cfg, device=self.device)
        else:
            return transform(magnitude, cfg=self.cfg)
    
    def _reorder_ops(self, ops, list):
        # Convert B to a set for faster lookup
        list_set = set(list)
        # Filter elements from A that are in B
        in_list = np.array([item for item in ops if item in list_set])
        # Filter elements from A that are not in B
        not_in_list = np.array([item for item in ops if item not in list_set])
        # Concatenate the arrays to get the desired order
        return np.concatenate((in_list, not_in_list))

    def random_augmentation(self, N, M):
        ops = self._sample_ops(N)
        ops = self._reorder_ops(ops, gan_transforms + diffusion_transforms)
        ops = self._add_magnitude(ops, M)
        transforms, _ = self._map_to_transforms(ops)
        return transforms
