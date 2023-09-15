import numpy as np
import torch
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from fvcore.transforms.transform import Transform
from randaug.data.transforms.corruptions import corrupt
from enum import Enum
            
class Augmentations(Enum):

    def __str__(self):
        return str(self.value)

    CYCLE_GAN = 'cycle_gan'
    CUT = 'cut'
    SAE = 'swapping_autoencoder'
    FOG = 'fog'
    RAIN = 'rain'
    SNOW = 'snow'
    DROP = 'drop'
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
    SAMPLE_PAIRING = 'sample_pairing'
        

# Own augmentations
class FogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.FOG
        self.magnitude = magnitude

    def get_transform(self, image):
        return WeatherTransform(name=self.name, severity=self.magnitude)
    

class RainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.RAIN
        self.magnitude = magnitude

    def get_transform(self, image):
        return WeatherTransform(name=self.name, severity=self.magnitude)


class SnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SNOW
        self.magnitude = magnitude

    def get_transform(self, image):
        return WeatherTransform(name=self.name, severity=self.magnitude)


class DropAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.DROP
        self.magnitude = magnitude

    def get_transform(self, image):
        return WeatherTransform(name=self.name, severity=self.magnitude)
    

class CycleGANAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CYCLE_GAN
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, severity=self.magnitude, file_name=file_name)


class CUTAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CUT
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, severity=self.magnitude, file_name=file_name)
    

class SAEAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SAE
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, severity=self.magnitude, file_name=file_name)
    

# Augmentations from paper: 
# AutoAugment: Learning Augmentation Strategies from Data

class ShearXAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SHEAR_X
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)    


class ShearYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SHEAR_Y
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)


class TranslateXAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.TRANSLATE_X
        self.magnitude = magnitude
        
    def get_transform(self, image): 
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)    


class TranslateYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.TRANSLATE_Y
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)    


class RotationAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.ROTATE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        h, w = image.shape[:2]
        angle = np.linspace(-30, 30, num=self.magnitude+1)
        return T.RotationTransform(h, w, angle[self.magnitude-1])
    

class AutoContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.AUTO_CONTRAST
        self.magnitude = magnitude
        
    def get_transform(self, image): # missing
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)    


class InvertAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.INVERT
        self.magnitude = magnitude
        
    def get_transform(self, image): # missing
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.PILColorTransform(func)


class EqualizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.EQUALIZE
        self.magnitude = magnitude
        
    def get_transform(self, image): # missing
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)    


class SolarizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SOLARIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0, 256, num=self.magnitude+1)
        func = lambda x: PIL.ImageOps.solarize(image, v[self.magnitude])
        return T.PILColorTransform(func)    


class PosterizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.POSTERIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(4, 8, num=self.magnitude+1)
        func = lambda x: PIL.ImageOps.posterize(x, v[self.magnitude])
        return T.PILColorTransform(func)


class ContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CONTRAST
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x: PIL.ImageEnhance.Contrast(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)    


class ColorAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.COLOR
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class BrightnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.BRIGHTNESS
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        print(v)
        func = lambda x: PIL.ImageEnhance.Brightness(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class SharpnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SHARPNESS
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x: PIL.ImageEnhance.Sharpness(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

# gemoetric?
class CutoutAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CUTOUT
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.0, 0.2, num=self.magnitude+1)
        func = lambda x: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)
    

# geometric?
class SamplePairingAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SAMPLE_PAIRING
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=self.magnitude+1)
        func = lambda x, v: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude-1])
        return T.ColorTransform(func)
    

# Wrapper class for random augmentations
class RandomAugmentation():
    
    def __init__(self, N, M, transforms):
        self.N = N
        self.M = M # list of magnitudes
        self.transforms = transforms # list of transforms
    
    def __repr__(self):
        if len(self.transforms):
            repr = '-'.join([f'{t.__class__.__name__}-{m}' for t, m in zip(self.transforms, self.M)])
        else:
            repr = "no-augmentation"
        return f'{self.N}-{repr}'
    
    def get_transforms(self):
        return self.transforms
    

    # rename Transforms
class WeatherTransform(Transform):

    def __init__(self, name, severity):
        super().__init__()
        self.name = name
        self.severity = severity

    def apply_image(self, img: np.ndarray):
        corr_img = corrupt(image=img, severity=self.severity, corruption_name=self.name)
        return corr_img
    
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    

class GANTransform(Transform):

    def __init__(self, network, severity, file_name):
        super().__init__()
        self.name = str(network)
        self.severity = severity
        self.file_name = file_name

    def apply_image(self, img: np.ndarray):
        path = self.file_name.split('/')
        path = '/'.join(path[:6]) + f'/augmentation/{self.name}/' + '/'.join(path[8:])
        return utils.read_image(path, format="BGR")
        
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    

