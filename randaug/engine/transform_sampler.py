import detectron2.data.transforms as T
import numpy as np
import sys
import random
from detectron2.data.transforms.augmentation_impl import *
from randaug.data.transforms.transforms import *
from fvcore.transforms.transform import NoOpTransform


from itertools import product

gan_transforms = [
    "CycleGANFogAugmentation",
    "CycleGANRainAugmentation",
    "CycleGANSnowAugmentation",
    "CUTFogAugmentation",
    "CUTRainAugmentation",
    "CUTSnowAugmentation",
]

diffusion_transforms = [
    "StableDiffusionFogAugmentation",
    "StableDiffusionRainAugmentation",
    "StableDiffusionSnowAugmentation",
    "CycleDiffusionFogAugmentation",
    "CycleDiffusionRainAugmentation",
    "CycleDiffusionSnowAugmentation",
    "MGIEDiffusionFogAugmentation",
    "MGIEDiffusionRainAugmentation",
    "MGIEDiffusionSnowAugmentation",
    "PlugPlayFogAugmentation",
    "PlugPlayRainAugmentation",
    "PlugPlaySnowAugmentation",
    "ControlNetFogAugmentation",
    "ControlNetRainAugmentation",
    "ControlNetSnowAugmentation",
    "ControllableSDFogAugmentation",
    "ControllableSDRainAugmentation",
    "ControllableSDSnowAugmentation"
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
    "AutoContrastAugmentation",
    "InvertAugmentation",
    "EqualizeAugmentation",
    "SolarizeAugmentation",
    "PosterizeAugmentation",
    "ContrastAugmentation",
    "BrightnessAugmentation",
    "SharpnessAugmentation",
    "Defocus",
    "ZoomBlur",
    "MotionBlur",
    "GlassBlur",
    "ShotNoise",
    "SaltAndPepper",
    "RandomSnow",
    "RandomRain",
    "RandomGravel", 
    "RandomGamma",
    "RandomFog",
    "PlasmaBrightnessContrast",
    "IsoNoise",
    "HueSaturationValue",
    "Emboss",
    "ColorJitter",
    "ChromaticAberration",
    "ChannelShuffle",
    "AdditiveNoise"
]

dropout_transforms = ["CutoutAugmentation", "ChannelDropout", "PixelDropout"]
weather_transforms = [ "FogAugmentation", "SnowAugmentation", "RainAugmentation", "DropAugmentation"]
autoaugment_transforms = ["AutoAugmentAugmentation", "RandAugmentAugmentation", "TrivialWideAugmentation", "AugMixAugmentation"]
geometric_transforms = ["ShearXAugmentation", "ShearYAugmentation", "TranslateXAugmentation", "TranslateYAugmentation", "RotationAugmentation"]

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
        transforms = gan_transforms + diffusion_transforms
        for transform in transforms:
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
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg, aug[0]) for aug in augs]
        print(f"Start from epoch: {self.epochs}")
        return augs[self.epochs:]
    
    def random_search(self):
        ops = self._sample_augmentation(magnitude=None)
        ops = self._filter_image_augmentations(ops)
        augs = [self._map_to_transforms(op) for op in ops]
        augs = [RandomAugmentation(self.cfg, aug[0]) for aug in augs]
        return augs
        
    def no_augmentation(self):
        return [RandomAugmentation(self.cfg, [])]
    
    def test(self, magnitude=0):
        #return [RandomAugmentation(self.cfg, 1, [])]
        #return [RandomAugmentation(self.cfg, 1, [CycleGANFogAugmentation(magnitude=1, cfg=self.cfg), ShearXAugmentation(magnitude=1, cfg=self.cfg), SnowAugmentation(magnitude=2, cfg=self.cfg)])]
        return [RandomAugmentation(self.cfg, [DropAugmentation(magnitude=magnitude, cfg=self.cfg)])]
        #return [RandomAugmentation(self.cfg, 1, [ComfyUIAugmentation(experiment="experiment_032_rain", cfg=self.cfg), RainAugmentation(magnitude=4, cfg=self.cfg)])]
    
    def diffusion_search(self):
        ops = self._sample_diffusion_models(ids=[0])
        augs = [self._map_to_transforms(op) for op in ops] 
        augs = [RandomAugmentation(self.cfg, aug[0]) for aug in augs]
        return augs
    
    def experiment(self, experiment: str):
        augs = [ComfyUIAugmentation(experiment=experiment, cfg=self.cfg)]
        augs = [RandomAugmentation(self.cfg, [aug]) for aug in augs]
        return augs
    
    def augmentation(self, augmentation: str, magnitude: int):
        aug_class = getattr(sys.modules[__name__], augmentation)
        aug = aug_class(magnitude=magnitude, cfg=self.cfg)
        return [RandomAugmentation(self.cfg, [aug])]
    
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
        augs = [RandomAugmentation(self.cfg, aug[0]) for aug in augs]
        return augs


class RandomSampler():

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.probability = cfg.aug_prob
        self.iterations = 0
    
    def adverse_augmentation(self):
        # magnitude 
        # progressive training
        # use deep augmentation
        # use traditional augmentation
        # use adverse overlay
        # define augmentation set
        magnitude = self.get_magnitude()
        use_deep = self.cfg.use_deep
        use_overlay = self.cfg.use_overlay
        use_standard = self.cfg.use_standard
        transforms = self.build_augmentation_stack(deep=use_deep, overlay=use_overlay, standard=use_standard, magnitude=magnitude)
        self.update_iterations()
        return transforms
    
    def build_augmentation_stack(self, deep=True, overlay=True, standard=True, magnitude=0):
        # build augmentation stack, sample from deep, sample from overlays, sample from traditional
        deep_augmentations = self.deep_augmentations(magnitude=magnitude, use_deep=deep)
        overlay_augmentations = self.overlay_augmentations(magnitude=magnitude, use_overlay=overlay)
        standard_augmentations = self.standard_augmentation(magnitude=magnitude, use_standard=standard)
        post_processing = self.postprocessing_augmentations()

        return deep_augmentations + overlay_augmentations + standard_augmentations + post_processing
    
    def deep_augmentations(self, magnitude=0, use_deep=True):
        if not use_deep:
            return []
        transforms = diffusion_transforms + gan_transforms
        return self._sample(transforms, magnitude)
    
    def overlay_augmentations(self, magnitude=0, use_overlay=True):
        if not use_overlay:
            return []
        transforms = weather_transforms
        return self._sample(transforms, magnitude)
    
    def standard_augmentation(self, magnitude=0, use_standard=True):
        if not use_standard:
            return []
        transforms = image_transforms + dropout_transforms + geometric_transforms
        return self._sample(transforms, magnitude)
    
    def postprocessing_augmentations(self):
        transforms = []
        transforms.append(T.RandomFlip(prob=0.5))
        if self.cfg.cutout_postprocessing:
            transforms.append(CutoutAugmentation(cfg=self.cfg))
        if self.cfg.box_postprocessing:
            transforms.append(BoundingboxAugmentation(algorithm='dino'))
        
        return transforms

    def update_iterations(self):
        self.iterations += (1 / 2.0) # num gpus
    
    def get_magnitude(self):
        if self.cfg.progressive:
            return min(math.floor((self.iterations / self.cfg.SOLVER.MAX_ITER) * self.cfg.magnitude), self.cfg.magnitude)
        else:
            return self.cfg.magnitude
        
    def _sample(self, transforms, magnitude):
        aug = random.sample(transforms, 1)[0]
        aug_class = getattr(sys.modules[__name__], aug)
        aug = self._init_transform(aug_class, magnitude)
        return [aug]
    
     # Random sampling causes the drop augmentation not to distribute across devices so this function is needed
    def _init_transform(self, transform, magnitude) -> T.Augmentation:
        if transform == DropAugmentation:
            return transform(magnitude, cfg=self.cfg, device=self.device)
        else:
            return transform(magnitude, cfg=self.cfg)
    
    # # old code
    # def _reorder_ops(self, ops, list):
    #     # Convert B to a set for faster lookup
    #     list_set = set(list)
    #     # Filter elements from A that are in B
    #     in_list = np.array([item for item in ops if item in list_set])
    #     # Filter elements from A that are not in B
    #     not_in_list = np.array([item for item in ops if item not in list_set])
    #     # Concatenate the arrays to get the desired order
    #     return np.concatenate((in_list, not_in_list))

    # def random_augmentation(self, N, M):
    #     if random.uniform(0, 1) > self.probability:
    #         return [NoOpTransform()]
    #     ops = self._sample_ops(N)
    #     ops = self._reorder_ops(ops, gan_transforms + diffusion_transforms)
    #     ops = self._add_magnitude(ops, M)
    #     transforms, _ = self._map_to_transforms(ops)
    #     transforms.append(T.RandomFlip(prob=0.5))

    #     if self.cfg.box_postprocessing == True:
    #         transforms.append(BoundingboxAugmentation(algorithm='dino'))

    #     if self.cfg.save_image == True:
    #         transforms.append(BoundingboxAugmentation(algorithm='generate_samples', cfg=self.cfg))

    #     return transforms
    
    # def _sample_ops(self, N):
    #     transforms = ai_transforms + image_transforms
    #     one_time = ai_transforms # ["DropAugmentation"]

    #     while True:
    #         # Sample elements from A
    #         sampled = np.random.choice(transforms, N) # type: ignore
    #         # Count how many elements are in B
    #         gan_count = sum(element in one_time for element in sampled)

    #         # It does not make sense to have more than one gan / diffusion transform as it replaces the rest of the augmentations
    #         if gan_count <= 1:
    #             sampled = [self._replace_sample(s) for s in sampled]
    #             return sampled
            
    # # This is implemented to reduce the amount of sampling for ai based augmentations
    # # 
    # def _replace_sample(self, sample):
    #     if sample in ai_transforms:
    #         while True:
    #             condition = random.choice(ai_conditions)
    #             aug = f"{sample}{condition}"
    #             if aug in diffusion_transforms + gan_transforms:
    #                 return aug 
    #     else:
    #         return sample
    
    # def _add_magnitude(self, ops, M):
    #     magnitude = M
    #     return [(o, magnitude) for o in ops]
    
    # def _map_to_transforms(self, ops):
    #     transforms = [getattr(sys.modules[__name__], op[0]) for op in ops]
    #     transforms = [self._init_transform(t, op[1]) for t, op in zip(transforms, ops)]
    #     magnitudes = [op[1] for op in ops]
    #     return transforms, magnitudes