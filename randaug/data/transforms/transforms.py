import numpy as np
import albumentations as A
import os
import math
import random
from detectron2.utils import comm
from PIL import ImageOps, ImageEnhance, Image
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from fvcore.transforms.transform import Transform
from randaug.data.transforms.corruptions import corrupt
from randaug.data.albumentations import AlbumentationsTransform, prepare_param
from enum import Enum
from randaug.data.transforms.box_transforms import SimpleBBTransform, AdjustBBTransform, SimilarityBBTransform, OutputBBTransform, CLIPBBTransform, DINOBBTransform, SaveTransform, CutoutBBTransform
from torchvision import transforms as v1

MAGNITUDE_BINS = 5 
DATA_PATH = "/mnt/ssd2/dataset/" 

class Augmentations(Enum):

    def __str__(self):
        return str(self.value)

    CYCLE_GAN = 'cyclegan'
    CUT = 'cut'
    CYCLE_DIFFUSION = 'cyclediffusion'
    STABLE_DIFFUSION = 'stablediffusion'
    MGIE_DIFFUSION = 'mgiediffusion'
    FOG = 'fog'
    RAIN = 'rain'
    SNOW = 'snow'
    DROP = 'drops'
    SHEAR_X = 'shear_x'
    SHEAR_Y = 'shear_y'
    TRANSLATE_X = 'translate_x'
    TRANSLATE_Y = 'translatye_y'
    ROTATE = 'rotate'
    AUTO_CONTRAST = 'auto_contrast'
    INVERT = 'invert'
    EQUALIZE = 'equalize'
    FLIP = 'flip'
    SOLARIZE = 'solarize'
    POSTERIZE = 'posterize'
    CONTRAST = 'contrast'
    COLOR = 'color'
    BRIGHTNESS = 'brightness'
    SHARPNESS = 'sharpness'
    CUTOUT = 'cutout'
    AUTO_AUGMENT = 'autoaugment'
    RAND_AUGMENT = 'randaugment'
    TRIVIAL_WIDE = 'trivialwide'
    AUG_MIX = 'augmix'

def random_invert(number):
    if random.choice([True, False]):
        return -number
    else:
        return number

# Own augmentations
class FogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=self.magnitude)
    

class RainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=self.magnitude)


class SnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=self.magnitude)


# run on GPU
class DropAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None, device=None):
        super().__init__()
        self.name = Augmentations.DROP
        self.magnitude = magnitude
        self.device = self._get_device(device)
        self.cfg = cfg

    def _get_device(self, device):
        if device is None:
            return f'cuda:{comm.get_rank()}'
        else:
            return device
        
    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=self.magnitude, device=self.device)
    

class CycleGANFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CYCLE_GAN
        self.weather = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)


class CycleGANRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CYCLE_GAN
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class CycleGANSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CYCLE_GAN
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class CUTFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CUT
        self.weather = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)


class CUTRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CUT
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class CUTSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CUT
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class CycleDiffusionFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CYCLE_DIFFUSION
        self.weather = Augmentations.FOG
        self.magnitude = magnitude#
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)


class CycleDiffusionRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CYCLE_DIFFUSION
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class CycleDiffusionSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CYCLE_DIFFUSION
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class StableDiffusionFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.STABLE_DIFFUSION
        self.weather = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class StableDiffusionRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.STABLE_DIFFUSION
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class StableDiffusionSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.STABLE_DIFFUSION
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    
class MGIEDiffusionFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.MGIE_DIFFUSION
        self.weather = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return MGIETransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class MGIEDiffusionRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.MGIE_DIFFUSION
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return MGIETransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class MGIEDiffusionSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.MGIE_DIFFUSION
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return MGIETransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class ComfyUIAugmentation(T.Augmentation):
    
    def __init__(self, experiment: str, cfg=None):
        super().__init__()
        self.name = experiment
        self.weather = experiment.split('_')[-1]
        self.cfg = cfg
        self.progressive_count = 0

    def get_transform(self, image, file_name):
        #return MGIETransform(network=self.name, weather=self.weather, severity=0, file_path=file_name, cfg=self.cfg)
        self.progressive_count += 1
        return ComfyUITransform(network=self.name, weather=self.weather, severity=self.progressive_count, file_path=file_name, cfg=self.cfg)
    


# Augmentations from paper: 
# AutoAugment: Learning Augmentation Strategies from Data

class ShearXAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SHEAR_X
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image):
        v = np.linspace(5, 25, num=MAGNITUDE_BINS) # Shear in degrees 
        v = random_invert(number=v)
        transform = A.Affine(shear={'x': v[self.magnitude], 'y': 0})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))


class ShearYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SHEAR_Y
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(5, 25, num=MAGNITUDE_BINS) # Shear in degrees 
        v = random_invert(number=v)
        transform = A.Affine(shear={'x': 0, 'y': v[self.magnitude]})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))


class TranslateXAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.TRANSLATE_X
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image): 
        v = np.linspace(0.1, 0.45, num=MAGNITUDE_BINS)
        v = random_invert(number=v)
        transform = A.Affine(translate_percent={'x': v[self.magnitude], 'y': 0})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))


class TranslateYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.TRANSLATE_Y
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 0.45, num=MAGNITUDE_BINS)
        v = random_invert(number=v)
        transform = A.Affine(translate_percent={'x': 0, 'y': v[self.magnitude]})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))  


# retrain (magnitude was -1)
class RotationAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.ROTATE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        angle = np.linspace(5, 30, num=MAGNITUDE_BINS)
        angle = random_invert(number=angle)
        transform = A.Affine(rotate=angle[self.magnitude])
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))  
        #return T.RotationTransform(h, w, angle[self.magnitude])
    

class AutoContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.AUTO_CONTRAST
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        func = lambda x: ImageOps.autocontrast(x)
        return T.PILColorTransform(func)    


class InvertAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.INVERT
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        func = lambda x: ImageOps.invert(x)
        return T.PILColorTransform(func)


class EqualizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.EQUALIZE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        func = lambda x: ImageOps.equalize(x)
        return T.PILColorTransform(func)    


class SolarizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SOLARIZE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(256, 0, num=MAGNITUDE_BINS)
        func = lambda x: ImageOps.solarize(x, v[self.magnitude])
        return T.PILColorTransform(func)


class PosterizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.POSTERIZE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(8, 1, num=MAGNITUDE_BINS)
        func = lambda x: ImageOps.posterize(x, int(v[self.magnitude]))
        return T.PILColorTransform(func)


class ContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTRAST
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Contrast(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)    


class ColorAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.COLOR
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.0, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Color(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class BrightnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.BRIGHTNESS
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Brightness(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class SharpnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SHARPNESS
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Sharpness(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class CutoutAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CUTOUT
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(1, 10, num=MAGNITUDE_BINS)
        m = int(v[self.magnitude])
        transform = A.CoarseDropout(min_holes=m, max_holes=m*8, 
                                    min_height=m, max_height=m*12, 
                                    min_width=m, max_width=m*12, p=1.0)
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=image.shape[:2])


class BoundingboxAugmentation(T.Augmentation):
    
    def __init__(self, algorithm=None, cfg=None):
        self.algorithm = algorithm
        self.cfg = cfg
    
    def get_transform(self, image, file_name, transforms):
        if self.algorithm == 'simple':
            return SimpleBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'adjust':
            return AdjustBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'similarity':
            return SimilarityBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'generate_sample_boxes':
            return OutputBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'generate_samples':
            return SaveTransform(image=image, file_name=file_name, transforms=transforms, path=self.cfg.save_path) # type: ignore
        elif self.algorithm == 'clip':
            return CLIPBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'dino':
            return DINOBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'cutout':
            return CutoutBBTransform(image=image, file_name=file_name, transforms=transforms)
        else:
            return NotImplementedError

class AutoAugmentAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.AUTO_AUGMENT
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        return AutoAugmentTransform(name=self.name, severity=self.magnitude)


class RandAugmentAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RAND_AUGMENT
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        return AutoAugmentTransform(name=self.name, severity=self.magnitude)
    
class TrivialWideAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.TRIVIAL_WIDE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        return AutoAugmentTransform(name=self.name, severity=self.magnitude)
    
class AugMixAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.AUG_MIX
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        return AutoAugmentTransform(name=self.name, severity=self.magnitude)

# Wrapper class for random augmentations
class RandomAugmentation():
    
    def __init__(self, cfg, M, augmentations):
        self.cfg = cfg
        self.M = M # list of magnitudes
        self.augmentations = augmentations # list of transforms
    
    def __repr__(self):
        if len(self.augmentations):
            repr = '-'.join([f'{t.__class__.__name__}-{m}' for t, m in zip(self.augmentations, self.M)])
        else:
            repr = "no-augmentation"
        return f'{self.cfg.rand_N}-{repr}'
    
    def get_transforms(self):
        return self.augmentations + self._append_standard_flip() + self._append_standard_transform()
    
    def _append_standard_flip(self):
        aug = T.RandomFlip(prob=0.5)
        return [aug]

    def _append_standard_transform(self):
        augs = []
        if self.cfg.cutout_postprocessing == True:
            augs.append(BoundingboxAugmentation(algorithm='cutout'))
        if self.cfg.box_postprocessing == True:
            augs.append(BoundingboxAugmentation(algorithm='dino'))
        return augs
    

class WeatherTransform(Transform):

    def __init__(self, name, severity, device=None):
        super().__init__()
        self.name = name
        self.severity = severity + 1
        self.device = device

    def apply_image(self, img: np.ndarray):
        corr_img = corrupt(image=img, severity=self.severity, corruption_name=self.name, device=self.device)
        return corr_img
    
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    

class GANTransform(Transform):

    def __init__(self, network, weather, severity, file_path, cfg):
        super().__init__()
        self.name = network
        self.weather = weather
        self.severity = severity
        self.file_path = file_path
        self.cfg = cfg
        self.image = self._read_image(self.file_path)
        self.original: np.ndarray

    def _read_image(self, file_path: str):
        filename = file_path.split('/')[-1]
        filename_jpg = f'{filename[:-4]}.jpg'
        path = os.path.join(f'{DATA_PATH}/cvpr24/adverse/augmentation', str(self.name), str(self.weather), filename_jpg)
        # path = os.path.join(f'{DATA_PATH}/cvpr24/adverse/itsc_augmentation', str(self.name), str(self.weather), filename_jpg)

        if not os.path.isfile(path):
            path = f'{path[:-4]}.png'
        return utils.read_image(path, format=self.cfg.INPUT.FORMAT)
    
    def apply_image(self, img: np.ndarray):
        self.original = img
        return self.image
        
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
class ComfyUITransform(Transform):

    def __init__(self, network, weather, severity, file_path, cfg):
        super().__init__()
        self.name = network
        self.weather = weather
        self.severity = severity
        self.file_path = file_path
        self.cfg = cfg
        self.image = self._read_image(self.file_path)
        self.original: np.ndarray

    def _read_image(self, file_path: str):
        filename = file_path.split('/')[-1]
        prefix = '' if not self.cfg.weather == 'diverse' else self.random_weather_prefix()
        filename_jpg = f'{prefix}{filename[:-4]}.jpg'
        
        if self.cfg.training == 'progressive':
            self.name = self.progressive_experiment()
        elif self.cfg.training == 'random':
            self.name = self.random_experiment()
        path = os.path.join(f'{DATA_PATH}/pami_train', str(self.name), filename_jpg)
        
        if not os.path.isfile(path):
            path = f'{path[:-4]}.png'
        return utils.read_image(path, format=self.cfg.INPUT.FORMAT)
    
    def apply_image(self, img: np.ndarray):
        self.original = img
        return self.image
        
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation

    def progressive_experiment(self):
        experiments = self.get_experiments()
        experiments.reverse()
        iterations = self.cfg.SOLVER.MAX_ITER
        progressive_count = max(self.severity, 0) / 2.0  # Ensure progressive_count is non-negative
        choice = min(math.floor((progressive_count / iterations) * len(experiments)), len(experiments) - 1)
        if self.cfg.weather == 'diverse':
            return f'experiment_{experiments[choice]}'
        else:
            return f'experiment_{experiments[choice]}_{self.cfg.weather}'
    
    def random_experiment(self):
        experiments = self.get_experiments()
        #experiments = ['024', '030', '031', '032', '033', '034']
        choice = np.random.choice(experiments)
        if self.cfg.weather == 'diverse':
            return f'experiment_{choice}'
        else:
            return f'experiment_{choice}_{self.cfg.weather}'
        
    def get_experiments(self):
        experiments = ['024', '030', '031', '032', '033', '034', '035', '036', '037', '038']
        if self.cfg.weather == 'fog':
            experiments = ['039', '040', '041', '042', '043', '044', '045', '046', '047', '048']
        return experiments

    def random_weather_prefix(self):
        # List of possible weather prefixes
        prefixes = ['fog_', 'rain_', 'snow_']
        
        # Randomly select and return one of the prefixes
        return random.choice(prefixes)

    

class MGIETransform(Transform):

    def __init__(self, network, weather, severity, file_path, cfg):
        super().__init__()
        self.name = network
        self.weather = weather
        self.severity = severity
        self.file_path = file_path
        self.cfg = cfg
        self.image = self._read_image2(self.file_path)
        self.original: np.ndarray
    
    def _read_image(self, file_path: str):
        filename = file_path.split('/')[-1]
        filename_jpg = f'{filename[:-4]}.jpg'
        path = os.path.join(f'{DATA_PATH}/cvpr24/adverse/augmentation', str(self.name), str(self.weather), filename_jpg)
        # path = os.path.join(f'{DATA_PATH}/cvpr24/adverse/itsc_augmentation', str(self.name), str(self.severity), str(self.weather), filename_jpg)
    
        if not os.path.isfile(path):
            path = f'{path[:-4]}.png'
        return utils.read_image(path, format=self.cfg.INPUT.FORMAT)
    
    def _read_image2(self, file_path: str):
        filename = file_path.split('/')[-1]
        prefix = self.random_weather_prefix()
        filename_jpg = f'{prefix}{filename[:-4]}.jpg'
        path = os.path.join(f'{DATA_PATH}/pami_train', str(self.name), filename_jpg)

        if not os.path.isfile(path):
            path = f'{path[:-4]}.png'
        return utils.read_image(path, format=self.cfg.INPUT.FORMAT)
    
    def random_weather_prefix(self):
        #return ''
        # List of possible weather prefixes
        prefixes = ['fog_', 'rain_', 'snow_']
        
        # Randomly select and return one of the prefixes
        return random.choice(prefixes)
    
    def apply_image(self, img: np.ndarray):
        self.original = img
        return self.image
        
    def apply_coords(self, coords):
        """
        Apply crop transform on coordinates.

        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """

        """
        Crop and resize coords
        """
        if self.original.shape == self.image.shape:
            return coords
        else:
            h, w = self.original.shape[0:2]
            h_new, w_new = self.image.shape[0:2]
            p = (w-h) // 2
            coords[:, 0] -= p
            coords[:, 0] = coords[:, 0] * (w_new * 1.0 / h)
            coords[:, 1] = coords[:, 1] * (h_new * 1.0 / h)
            return coords

    
    def apply_segmentation(self, segmentation):
        return segmentation
    

class AutoAugmentTransform(Transform):

    def __init__(self, name, severity, device=None):
        super().__init__()
        self.name = name
        self.device = device
        self.severity = severity
        self.transform = self.setup_augmentation(self.name)

    def setup_augmentation(self, policy):
        if policy == Augmentations.AUTO_AUGMENT:
            return v1.AutoAugment(policy=v1.AutoAugmentPolicy.IMAGENET)
        elif policy == Augmentations.RAND_AUGMENT:
            return v1.RandAugment(num_ops=2)
        elif policy == Augmentations.AUG_MIX:
            return v1.AugMix()
        elif policy == Augmentations.TRIVIAL_WIDE:
            return v1.TrivialAugmentWide()
        else:
            return NotImplementedError


    def apply_image(self, img: np.ndarray):
        img = self.transform(Image.fromarray(img)) # type: ignore
        return np.array(img)
    
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation