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
from randaug.data.albumentations import AlbumentationsTransform
from enum import Enum
from randaug.data.transforms.box_transforms import SimpleBBTransform, AdjustBBTransform, SimilarityBBTransform, OutputBBTransform, CLIPBBTransform, DINOBBTransform, SaveTransform, CutoutBBTransform
from torchvision import transforms as v1
import cv2

MAGNITUDE_BINS = 10 
DATA_PATH = "/home/rothmeier/Documents/datasets/" 

class Augmentations(Enum):

    def __str__(self):
        return str(self.value)

    CYCLE_GAN = 'cyclegan'
    CUT = 'cut'
    CYCLE_DIFFUSION = 'cycle_diffusion'
    STABLE_DIFFUSION = 'stable_diffusion'
    MGIE_DIFFUSION = 'mgie_diffusion'
    PLUG_DIFFUSION = 'plugplay_diffusion'
    CONTROL_DIFFUSION = 'control_diffusion'
    CONTROLLABLE_SD = 'controllable_sd'
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
    CHANNEL_DROPOUT = 'channel_dropout'
    DEFOCUS = 'defocus'
    ZOOM_BLUR = 'zoom_blur'
    ADDITIVE_NOISE = 'additive_noise'
    CHANNEL_SHUFFLE = 'channel_shuffle'
    CHROMATIC_ABERRATION = 'chromatic_aberration'
    COLOR_JITTER = 'color_jitter'
    EMBOSS = 'emboss'
    HUE_SATURATION_VALUE = 'hue_saturation_value'
    ISO_NOISE = 'iso_noise'
    PIXEL_DROPOUT = 'pixel_dropout'
    PLASMA_BRIGHTNESS_CONTRAST = 'plasma_brightness_contrast'
    RANDOM_FOG = 'random_fog'
    RANDOM_GAMMA = 'random_gamma'
    RANDOM_RAIN = 'random_rain'
    RANDOM_SNOW = 'random_snow'
    RANDOM_GRAVEL = 'random_gravel'
    SALT_AND_PEPPER = 'salt_and_pepper'
    SHOT_NOISE = 'shot_noise'
    GLASS_BLUR = 'glass_blur'
    MOTION_BLUR = 'motion_blur'
    GAUSSIAN_NOISE = 'gaussian_noise'


def random_invert(number):
    if random.choice([True, False]):
        return -number
    else:
        return number

def sample_magnitude(magnitude, lower_bound, upper_bound, invert=True, fixed=False, num_bins=10):
    v = np.linspace(lower_bound, upper_bound, num_bins)
    max_value = v[magnitude]  # upper bound from v based on magnitude
    
    if fixed:
        value = max_value if not invert else random_invert(max_value)
        return value
    else:
        value = random.uniform(lower_bound, max_value)
        value = value if not invert else random_invert(value)
        return value

# Own augmentations
class FogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=1, upper_bound=10, fixed=self.cfg.magnitude_fixed, invert=False) # type: ignore
        return WeatherTransform(name=str(self.name), severity=int(v))
    

class RainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=1, upper_bound=10, fixed=self.cfg.magnitude_fixed, invert=False)
        return WeatherTransform(name=str(self.name), severity=int(v))


class SnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=1, upper_bound=10, fixed=self.cfg.magnitude_fixed, invert=False)
        return WeatherTransform(name=str(self.name), severity=int(v))


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
        v = sample_magnitude(self.magnitude, lower_bound=1, upper_bound=10, fixed=self.cfg.magnitude_fixed, invert=False)
        return WeatherTransform(name=str(self.name), severity=int(v), device=self.device)
    

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
    
class PlugPlayFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.PLUG_DIFFUSION
        self.weather = Augmentations.FOG
        self.magnitude = magnitude#
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class PlugPlayRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.PLUG_DIFFUSION
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class PlugPlaySnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.PLUG_DIFFUSION
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)

class ControlNetFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTROL_DIFFUSION
        self.weather = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class ControlNetRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTROL_DIFFUSION
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class ControlNetSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTROL_DIFFUSION
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name, cfg=self.cfg)
    

class ControllableSDFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTROLLABLE_SD
        self.weather = Augmentations.FOG
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        v = sample_magnitude(self.magnitude, lower_bound=0, upper_bound=9, fixed=self.cfg.magnitude_fixed, invert=False) # type: ignore
        return ControllableSDTransform(network=self.name, weather=self.weather, severity=int(v), file_path=file_name, cfg=self.cfg)
    

class ControllableSDRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTROLLABLE_SD
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        v = sample_magnitude(self.magnitude, lower_bound=0, upper_bound=9, fixed=self.cfg.magnitude_fixed, invert=False) # type: ignore
        return ControllableSDTransform(network=self.name, weather=self.weather, severity=int(v), file_path=file_name, cfg=self.cfg)
    

class ControllableSDSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTROLLABLE_SD
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image, file_name):
        v = sample_magnitude(self.magnitude, lower_bound=0, upper_bound=9, fixed=self.cfg.magnitude_fixed, invert=False) # type: ignore
        return ControllableSDTransform(network=self.name, weather=self.weather, severity=int(v), file_path=file_name, cfg=self.cfg)
    

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
        v = np.linspace(3, 30, num=MAGNITUDE_BINS) # Shear in degrees 
        v = random_invert(number=v)
        if self.cfg.magnitude_fixed:
            transform = A.Affine(shear={"x": (v[self.magnitude], v[self.magnitude]), "y": (0, 0)}, p=1.0)
        else:
            transform = A.Affine(shear={"x": (0, v[self.magnitude]), "y": (0, 0)}, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class ShearYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SHEAR_Y
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(3, 30, num=MAGNITUDE_BINS) # Shear in degrees 
        v = random_invert(number=v)
        if self.cfg.magnitude_fixed:
            transform = A.Affine(shear={"x": (0, 0), "y": (v[self.magnitude], v[self.magnitude])}, p=1.0)
        else:
            transform = A.Affine(shear={"x": (0, 0), "y": (0, v[self.magnitude])}, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class TranslateXAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.TRANSLATE_X
        self.magnitude = magnitude
        self.cfg = cfg

    def get_transform(self, image): 
        v = np.linspace(0.1, 0.45, num=MAGNITUDE_BINS)
        v = random_invert(number=v)
        if self.cfg.magnitude_fixed:
            transform = A.Affine(translate_percent={"x": (v[self.magnitude], v[self.magnitude]), "y": (0, 0)}, p=1.0)
        else:
            transform = A.Affine(translate_percent={"x": (0, v[self.magnitude]), "y": (0, 0)}, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class TranslateYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.TRANSLATE_Y
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 0.45, num=MAGNITUDE_BINS)
        v = random_invert(number=v)
        if self.cfg.magnitude_fixed:
            transform = A.Affine(translate_percent={"x": (0, 0), "y": (v[self.magnitude], v[self.magnitude])}, p=1.0)
        else:
            transform = A.Affine(translate_percent={"x": (0, 0), "y": (0, v[self.magnitude])}, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])

class RotationAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.ROTATE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        angle = np.linspace(3, 30, num=MAGNITUDE_BINS)
        angle = random_invert(number=angle)
        if self.cfg.magnitude_fixed:
            transform = A.Affine(rotate=(angle[self.magnitude], angle[self.magnitude]), p=1.0)
        else:
            transform = A.Affine(rotate=(1, angle[self.magnitude]), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])  
        

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
        v = sample_magnitude(self.magnitude, lower_bound=230, upper_bound=0, fixed=self.cfg.magnitude_fixed, invert=False)
        func = lambda x: ImageOps.solarize(x, int(v))
        return T.PILColorTransform(func)


class PosterizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.POSTERIZE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=8, upper_bound=1, fixed=self.cfg.magnitude_fixed, invert=False)
        func = lambda x: ImageOps.posterize(x, int(v))
        return T.PILColorTransform(func)


class ContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CONTRAST
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=0.1, upper_bound=1, fixed=self.cfg.magnitude_fixed, invert=True)
        func = lambda x: ImageEnhance.Contrast(x).enhance(1 + v)
        return T.PILColorTransform(func)    


class ColorAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.COLOR
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=0.1, upper_bound=1, fixed=self.cfg.magnitude_fixed, invert=False)
        func = lambda x: ImageEnhance.Color(x).enhance(1 - v)
        return T.PILColorTransform(func)
    

class BrightnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.BRIGHTNESS
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=0.1, upper_bound=1, fixed=self.cfg.magnitude_fixed, invert=True)
        func = lambda x: ImageEnhance.Brightness(x).enhance(1 + v)
        return T.PILColorTransform(func)
    

class SharpnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SHARPNESS
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=1.0, upper_bound=2.0, fixed=self.cfg.magnitude_fixed, invert=False)
        func = lambda x: ImageEnhance.Sharpness(x).enhance(v)
        return T.PILColorTransform(func)
    

class CutoutAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CUTOUT
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=1, upper_bound=10, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.CoarseDropout(num_holes_range=(1, int(v)), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


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
    
    def __init__(self, cfg, augmentations):
        self.cfg = cfg
        self.augmentations = augmentations # list of transforms
    
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
        self.severity = severity
        self.device = device

    def apply_image(self, img: np.ndarray):
        corr_img = corrupt(image=img, severity=self.severity, corruption_name=self.name, device=self.device)
        return corr_img
    
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    

class ImageReplaceTransform(Transform):
    def __init__(self, network, weather, severity, file_path, cfg, base_path):
        super().__init__()
        self.network = network
        self.weather = weather
        self.severity = severity
        self.file_path = file_path
        self.cfg = cfg
        self.base_path = base_path
        self.image = self._read_image()
        self.original: np.ndarray

    def _read_image(self):
        filename = os.path.basename(self.file_path)
        filename_jpg = f'{filename[:-4]}.jpg'
        
        path = os.path.join(self.base_path, str(self.network), '0', str(self.weather), filename_jpg)

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

# Now specialize this class:

class GANTransform(ImageReplaceTransform):
    def __init__(self, network, weather, severity, file_path, cfg):
        base_path = f'{DATA_PATH}/cvpr24/adverse/augmentation'
        # or: base_path = f'{DATA_PATH}/cvpr24/adverse/itsc_augmentation'
        super().__init__(network, weather, severity, file_path, cfg, base_path)

class ControllableSDTransform(ImageReplaceTransform):
    def __init__(self, network, weather, severity, file_path, cfg):
        base_path = f'{DATA_PATH}/pami_train'
        super().__init__(network, weather, severity, file_path, cfg, base_path)

    def _read_image(self):
        filename = os.path.basename(self.file_path)
        filename_jpg = f'{filename[:-4]}.jpg'
        experiments = self.get_experiments()
        experiments.reverse()

        print(self.severity)
        path = os.path.join(self.base_path, f'experiment_{experiments[self.severity]}_{self.weather}', filename_jpg)

        if not os.path.isfile(path):
            path = f'{path[:-4]}.png'

        return utils.read_image(path, format=self.cfg.INPUT.FORMAT)
    
    def get_experiments(self):
        experiments = ['024', '030', '031', '032', '033', '034', '035', '036', '037', '038']
        if self.weather == Augmentations.FOG:
            experiments = ['039', '040', '041', '042', '043', '044', '045', '046', '047', '048']
        return experiments
    
    
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
        self.image = self._read_image(self.file_path)
        self.original: np.ndarray
    
    def _read_image(self, file_path: str):
        filename = file_path.split('/')[-1]
        filename_jpg = f'{filename[:-4]}.jpg'
        path = os.path.join(f'{DATA_PATH}/cvpr24/adverse/augmentation', str(self.name), '0', str(self.weather), filename_jpg)
        # path = os.path.join(f'{DATA_PATH}/cvpr24/adverse/itsc_augmentation', str(self.name), str(self.severity), str(self.weather), filename_jpg)
    
        if not os.path.isfile(path):
            path = f'{path[:-4]}.png'
        return utils.read_image(path, format=self.cfg.INPUT.FORMAT)
    
    def _read_image2(self, file_path: str):
        filename = file_path.split('/')[-1]
        prefix = "" # self.random_weather_prefix()
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
    


# Albumentations

class AdditiveNoise(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.ADDITIVE_NOISE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        std = np.linspace(0.1, 0.5, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            transform = A.AdditiveNoise(noise_type="gaussian", spatial_mode="shared", 
                                    noise_params={"mean_range": (0.0, 0.0), "std_range": (std[self.magnitude], std[self.magnitude])}, 
                                    p=1.0)
        else:
            transform = A.AdditiveNoise(noise_type="gaussian", spatial_mode="shared", 
                                        noise_params={"mean_range": (0.0, 0.0), "std_range": (0.05, std[self.magnitude])}, 
                                        p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])
    
class ChannelShuffle(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CHANNEL_SHUFFLE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        transform = A.ChannelShuffle(p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])

class ChromaticAberration(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CHROMATIC_ABERRATION
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.05, 5, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            transform = A.ChromaticAberration(primary_distortion_limit=(v[self.magnitude], v[self.magnitude]), 
                                              secondary_distortion_limit=(-0.05, 0.05), mode='random', 
                                              interpolation=cv2.INTER_LINEAR, p=1.0)
        else:
            transform = A.ChromaticAberration(primary_distortion_limit=(0.0, v[self.magnitude]), 
                                              secondary_distortion_limit=(-0.05, 0.05), mode='random', 
                                              interpolation=cv2.INTER_LINEAR, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])

class ColorJitter(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.COLOR_JITTER
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            v = random_invert(number=v)
            transform = A.ColorJitter(brightness=(1 + v[self.magnitude], 1 + v[self.magnitude]),
                                      contrast=(1 + v[self.magnitude], 1 + v[self.magnitude]), 
                                      saturation=(1 + v[self.magnitude], 1 + v[self.magnitude]), p=1.0)
        else:
            transform = A.ColorJitter(brightness=v[self.magnitude], 
                                      contrast=v[self.magnitude], 
                                      saturation=v[self.magnitude], p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])

class Emboss(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.EMBOSS
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            transform = A.Emboss(alpha=(v[self.magnitude], v[self.magnitude]), strength=(v[self.magnitude], v[self.magnitude]), p=1.0)
        else:
            transform = A.Emboss(alpha=(0.05, v[self.magnitude]), strength=(0.05, v[self.magnitude]), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])
    
class HueSaturationValue(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.HUE_SATURATION_VALUE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(10, 100, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            v = random_invert(number=v)
            transform = A.HueSaturationValue(hue_shift_limit=(v[self.magnitude], v[self.magnitude]), 
                                             sat_shift_limit=(v[self.magnitude], v[self.magnitude]), 
                                             val_shift_limit=(v[self.magnitude], v[self.magnitude]), p=1.0)
        else:
            transform = A.HueSaturationValue(hue_shift_limit=v[self.magnitude], 
                                         sat_shift_limit=v[self.magnitude], 
                                         val_shift_limit=v[self.magnitude], p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])
    
class IsoNoise(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.ISO_NOISE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.0, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            transform = A.ISONoise(color_shift=(0.01, 1), intensity=(v[self.magnitude], v[self.magnitude]), p=1.0)
        else:
            transform = A.ISONoise(color_shift=(0.01, 1), intensity=(0.1, v[self.magnitude]), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])
    
# checkpoint

class PixelDropout(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.PIXEL_DROPOUT
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=0.1, upper_bound=0.9, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.PixelDropout(dropout_prob=v, per_channel=True, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])
    

class PlasmaBrightnessContrast(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.PLASMA_BRIGHTNESS_CONTRAST
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.0, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            v = random_invert(number=v)
            transform = A.PlasmaBrightnessContrast(brightness_range=(v[self.magnitude], v[self.magnitude]), 
                                                   contrast_range=(v[self.magnitude], v[self.magnitude]), 
                                                   roughness=5, p=1.0)
        else:
            transform = A.PlasmaBrightnessContrast(brightness_range=(-v[self.magnitude], v[self.magnitude]), 
                                                   contrast_range=(-v[self.magnitude], v[self.magnitude]), 
                                                   roughness=5, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])
    

class RandomFog(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RANDOM_FOG
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.0, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            transform = A.RandomFog(alpha_coef=0.08, fog_coef_range=(v[self.magnitude], v[self.magnitude]), p=1.0)
        else:
            transform = A.RandomFog(alpha_coef=0.08, fog_coef_range=(0.05, v[self.magnitude]), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class RandomGamma(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RANDOM_GAMMA
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=10, upper_bound=100, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.RandomGamma(gamma_limit=(80 + v, 120), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])
    

class RandomGravel(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RANDOM_GRAVEL
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=5, upper_bound=50, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.RandomGravel(gravel_roi=(0.0, 0.0, 1.0, 1.0), number_of_patches=int(v), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class RandomRain(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RANDOM_RAIN
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=1, upper_bound=10, fixed=self.cfg.magnitude_fixed, invert=False)
        rain_type = 'default'
        if v <= 2:
            rain_type = 'drizzle'
        elif v <= 5:
            rain_type = 'default'
        elif v <= 8:
            rain_type = 'heavy'
        else:
            rain_type = 'torrential'
        
        transform = A.RandomRain(rain_type=rain_type, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class RandomSnow(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.RANDOM_SNOW
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            transform = A.RandomSnow(snow_point_range=(v[self.magnitude], v[self.magnitude]), p=1.0)
        else:
            transform = A.RandomSnow(snow_point_range=(v[self.magnitude], 1.0), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class SaltAndPepper(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SALT_AND_PEPPER
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=0.01, upper_bound=0.25, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.SaltAndPepper(amount=(0.01, v), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class ShotNoise(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.SHOT_NOISE
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=0.05, upper_bound=0.5, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.ShotNoise(scale_range=(0.01, v), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class GlassBlur(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.GLASS_BLUR
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=0.1, upper_bound=1.0, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.GlassBlur(sigma=v, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class MotionBlur(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.MOTION_BLUR
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = np.linspace(5, 25, MAGNITUDE_BINS)
        if self.cfg.magnitude_fixed:
            transform = A.MotionBlur(blur_limit=(int(v[self.magnitude]), int(v[self.magnitude])), p=1.0)
        else:
            transform = A.MotionBlur(blur_limit=(3, int(v[self.magnitude])), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])



class ZoomBlur(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.ZOOM_BLUR
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=1.03, upper_bound=1.3, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.ZoomBlur(max_factor=(1, v), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class Defocus(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.DEFOCUS
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        v = sample_magnitude(self.magnitude, lower_bound=5, upper_bound=15, fixed=self.cfg.magnitude_fixed, invert=False)
        transform = A.Defocus(radius=(3, int(v)), p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])


class ChannelDropout(T.Augmentation):
     
    def __init__(self, magnitude=1, cfg=None):
        super().__init__()
        self.name = Augmentations.CHANNEL_DROPOUT
        self.magnitude = magnitude
        self.cfg = cfg
        
    def get_transform(self, image):
        transform = A.ChannelDropout(channel_drop_range=(1, 1), fill=0, p=1.0)
        return AlbumentationsTransform(transform, size=image.shape[:2])