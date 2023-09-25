import numpy as np
import torch
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import albumentations as A
from PIL import Image
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from fvcore.transforms.transform import Transform, TransformList
from randaug.data.transforms.corruptions import corrupt
from randaug.data.albumentations import AlbumentationsTransform, prepare_param
from enum import Enum

MAGNITUDE_BINS = 11 

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
        v = np.linspace(-20, 20, num=MAGNITUDE_BINS) # Shear in degrees 
        transform = A.Affine(shear={'x': v[self.magnitude], 'y': 0})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))


class ShearYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SHEAR_Y
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(-20, 20, num=MAGNITUDE_BINS) # Shear in degrees 
        transform = A.Affine(shear={'x': 0, 'y': v[self.magnitude]})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))

class TranslateXAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.TRANSLATE_X
        self.magnitude = magnitude
        
    def get_transform(self, image): 
        v = np.linspace(-0.45, 0.45, num=MAGNITUDE_BINS)
        transform = A.Affine(translate_percent={'x': v[self.magnitude], 'y': 0})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))


class TranslateYAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.TRANSLATE_Y
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(-0.45, 0.45, num=MAGNITUDE_BINS)
        transform = A.Affine(translate_percent={'x': 0, 'y': v[self.magnitude]})
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=(image.shape[:2]))  


class RotationAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.ROTATE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        h, w = image.shape[:2]
        angle = np.linspace(-30, 30, num=MAGNITUDE_BINS)
        return T.RotationTransform(h, w, angle[self.magnitude-1])
    

class AutoContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.AUTO_CONTRAST
        self.magnitude = magnitude
        
    def get_transform(self, image):
        func = lambda x: PIL.ImageOps.autocontrast(x)
        return T.PILColorTransform(func)    


class InvertAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.INVERT
        self.magnitude = magnitude
        
    def get_transform(self, image):
        func = lambda x: PIL.ImageOps.invert(x)
        return T.PILColorTransform(func)


class EqualizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.EQUALIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        func = lambda x: PIL.ImageOps.equalize(x)
        return T.PILColorTransform(func)    


class SolarizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SOLARIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(256, 0, num=MAGNITUDE_BINS)
        func = lambda x: PIL.ImageOps.solarize(x, v[self.magnitude])
        return T.PILColorTransform(func)


class PosterizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.POSTERIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(8, 1, num=MAGNITUDE_BINS)
        func = lambda x: PIL.ImageOps.posterize(x, int(v[self.magnitude]))
        return T.PILColorTransform(func)


class ContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CONTRAST
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: PIL.ImageEnhance.Contrast(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)    


class ColorAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.COLOR
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: PIL.ImageEnhance.Color(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class BrightnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.BRIGHTNESS
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: PIL.ImageEnhance.Brightness(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class SharpnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SHARPNESS
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: PIL.ImageEnhance.Sharpness(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class CutoutAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CUTOUT
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0, 10, num=MAGNITUDE_BINS)
        m = int(v[self.magnitude])
        transform = A.CoarseDropout(min_holes=m, max_holes=m*8, 
                                    min_height=m, max_height=m*12, 
                                    min_width=m, max_width=m*12, p=1.0)
        params = prepare_param(transform, image)
        return AlbumentationsTransform(transform, params, size=image.shape[:2])


class BoundingboxAugmentation(T.Augmentation):
    
    def __init__(self, algorithm=None):
        self.algorithm = algorithm
    
    def get_transform(self, image, file_name, transforms):
        if self.algorithm == 'invalidate':
            return InvalidateBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'adjust':
            return AdjustBBTransform(image=image, file_name=file_name, transforms=transforms)
        else:
            return NotImplementedError
    

# Wrapper class for random augmentations
class RandomAugmentation():
    
    def __init__(self, N, M, augmentations):
        self.N = N
        self.M = M # list of magnitudes
        self.augmentations = augmentations # list of transforms
    
    def __repr__(self):
        if len(self.augmentations):
            repr = '-'.join([f'{t.__class__.__name__}-{m}' for t, m in zip(self.augmentations, self.M)])
        else:
            repr = "no-augmentation"
        return f'{self.N}-{repr}'
    
    def get_transforms(self):
        arr = self._prepend_standard_transform() + self.augmentations + self._append_standard_transform()
        print(arr)
        return self._prepend_standard_transform() + self.augmentations + self._append_standard_transform()
    
    def _prepend_standard_transform(self):
        aug = T.RandomFlip(prob=0.5)
        return [aug]

    def _append_standard_transform(self):
        aug = BoundingboxAugmentation(algorithm='invalidate')
        return [aug]
    

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
    

# Bounding box transforms

# Invalidate bounding box depending on difference to orignal rectangle
class InvalidateBBTransform(Transform):

    def __init__(self, image: np.ndarray, file_name: str, transforms: list):
        super().__init__()
        self.image = image
        self.file_name = file_name
        self.previous = TransformList(transforms) # previous transforms

    def apply_image(self, img: np.ndarray):
        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            img = utils.read_image(self.file_name, format="BGR")
            transformed = self.previous.apply_image(img)
            return self._invalidate_bbox()
        except (AttributeError, NotImplementedError):
            return box  
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _invalidate_bbox(self):
        return np.array([np.Infinity,
                         np.Infinity,
                         np.Infinity,
                         np.Infinity])
    

# adjust bounding boxes according to what part of the vehicle can be seen
class AdjustBBTransform(Transform):

    def __init__(self, image: np.ndarray, file_name: str, transforms: list):
        super().__init__()
        self.image = image # transformed image
        self.file_name = file_name
        self.transforms = transforms # previous transforms
        self.previous = TransformList(transforms) # previous transforms

    def apply_image(self, img: np.ndarray):
        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            img = utils.read_image(self.file_name, format="BGR")
            transformed = self.previous.apply_image(img)
            return self._invalidate_bbox()        
        except (AttributeError, NotImplementedError):
            return box
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _invalidate_bbox(self):
        return np.array([np.Infinity,
                         np.Infinity,
                         np.Infinity,
                         np.Infinity])


# trained with classificator on images in bounding boxes
class SimpleBBTransform(Transform):

    def __init__(self, image: np.ndarray):
        super().__init__()
        self.image = image # transformed image

    def apply_image(self, img: np.ndarray):
        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            return self._invalidate_bbox()        
        except (AttributeError, NotImplementedError):
            return box
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _invalidate_bbox(self):
        return np.array([np.Infinity,
                         np.Infinity,
                         np.Infinity,
                         np.Infinity])