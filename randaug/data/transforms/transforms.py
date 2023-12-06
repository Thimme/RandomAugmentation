import numpy as np
import torch
import albumentations as A
import torchvision.transforms
import cv2
import os
import math
import uuid
import lpips
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt 
from detectron2.utils import comm
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from fvcore.transforms.transform import Transform, TransformList, NoOpTransform
from randaug.data.transforms.corruptions import corrupt
from randaug.data.albumentations import AlbumentationsTransform, prepare_param
from randaug.data.classifier.classifier import SimpleClassifier
from enum import Enum

MAGNITUDE_BINS = 5 

class Augmentations(Enum):

    def __str__(self):
        return str(self.value)

    CYCLE_GAN = 'cyclegan'
    CUT = 'cut'
    CYCLE_DIFFUSION = 'cyclediffusion'
    STABLE_DIFFUSION = 'stablediffusion'
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


# Own augmentations
class FogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.FOG
        self.magnitude = magnitude

    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=math.floor((self.magnitude+1)/2))
    

class RainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.RAIN
        self.magnitude = magnitude

    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=math.floor((self.magnitude+1)/2))


class SnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SNOW
        self.magnitude = magnitude

    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=math.floor((self.magnitude+1)/2))


# run on GPU
class DropAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.DROP
        self.magnitude = magnitude
        self.device = f'cuda:{comm.get_rank()}'

    def get_transform(self, image):
        return WeatherTransform(name=str(self.name), severity=math.floor((self.magnitude+1)/2), device=self.device)
    

class CycleGANFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CYCLE_GAN
        self.weather = Augmentations.FOG
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)


class CycleGANRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CYCLE_GAN
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class CycleGANSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CYCLE_GAN
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class CUTFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CUT
        self.weather = Augmentations.FOG
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)


class CUTRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CUT
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class CUTSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CUT
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class CycleDiffusionFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CYCLE_DIFFUSION
        self.weather = Augmentations.FOG
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)


class CycleDiffusionRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CYCLE_DIFFUSION
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class CycleDiffusionSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CYCLE_DIFFUSION
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class StableDiffusionFogAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.STABLE_DIFFUSION
        self.weather = Augmentations.FOG
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class StableDiffusionRainAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.STABLE_DIFFUSION
        self.weather = Augmentations.RAIN
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

class StableDiffusionSnowAugmentation(T.Augmentation):

    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.STABLE_DIFFUSION
        self.weather = Augmentations.SNOW
        self.magnitude = magnitude

    def get_transform(self, image, file_name):
        return GANTransform(network=self.name, weather=self.weather, severity=self.magnitude, file_path=file_name)
    

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
        func = lambda x: ImageOps.autocontrast(x)
        return T.PILColorTransform(func)    


class InvertAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.INVERT
        self.magnitude = magnitude
        
    def get_transform(self, image):
        func = lambda x: ImageOps.invert(x)
        return T.PILColorTransform(func)


class EqualizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.EQUALIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        func = lambda x: ImageOps.equalize(x)
        return T.PILColorTransform(func)    


class SolarizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SOLARIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(256, 0, num=MAGNITUDE_BINS)
        func = lambda x: ImageOps.solarize(x, v[self.magnitude])
        return T.PILColorTransform(func)


class PosterizeAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.POSTERIZE
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(8, 1, num=MAGNITUDE_BINS)
        func = lambda x: ImageOps.posterize(x, int(v[self.magnitude]))
        return T.PILColorTransform(func)


class ContrastAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.CONTRAST
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Contrast(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)    


class ColorAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.COLOR
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Color(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class BrightnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.BRIGHTNESS
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Brightness(x).enhance(v[self.magnitude])
        return T.PILColorTransform(func)
    

class SharpnessAugmentation(T.Augmentation):
     
    def __init__(self, magnitude=1):
        super().__init__()
        self.name = Augmentations.SHARPNESS
        self.magnitude = magnitude
        
    def get_transform(self, image):
        v = np.linspace(0.1, 1.9, num=MAGNITUDE_BINS)
        func = lambda x: ImageEnhance.Sharpness(x).enhance(v[self.magnitude])
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
            return SimpleBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'adjust':
            return AdjustBBTransform(image=image, file_name=file_name, transforms=transforms)
        elif self.algorithm == 'similarity':
            return SimilarityBBTransform(image=image, file_name=file_name, transforms=transforms)
        else:
            return NotImplementedError
    

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
        if self.cfg.box_postprocessing == True:
            return self.augmentations + self._append_standard_flip() + self._append_standard_transform()
        else:
            return self.augmentations + self._append_standard_flip()
    
    def _append_standard_flip(self):
        aug = T.RandomFlip(prob=0.5)
        return [aug]

    def _append_standard_transform(self):
        aug = BoundingboxAugmentation(algorithm='similarity')
        return [aug]
    

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
    

class GANTransform(Transform):

    def __init__(self, network, weather, severity, file_path):
        super().__init__()
        self.name = network
        self.weather = weather
        self.severity = severity
        self.file_path = file_path

    def apply_image(self, img: np.ndarray):
        filename = self.file_path.split('/')[-1]
        filename_jpg = f'{filename[:-4]}.jpg'
        path = os.path.join('/mnt/ssd2/dataset/cvpr24/adverse/augmentation', str(self.name), str(self.weather), filename_jpg)
        if not os.path.isfile(path):
            path = f'{path[:-4]}.png'
        return utils.read_image(path, format="BGR")
        
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    

# Bounding box transforms

def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

# Invalidate bounding box depending on difference to orignal rectangle
class SimilarityBBTransform(Transform):

    def __init__(self, image: np.ndarray, file_name: str, transforms: list):
        super().__init__()
        self.image = image # transformed image
        self.file_name = file_name
        self.original = np.array(Image.open(self.file_name))
        self.transforms = transforms[-1] # previous transforms
        self.original = self._apply_transforms(self.original, self.transforms)
        self.image = utils.convert_image_to_rgb(self.image, 'BGR')
        self.device = f'cuda:{comm.get_rank()}'
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device)
        self.count = 0

    def apply_image(self, img: np.ndarray):

        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            if self._compare_ssim(box=box) > 0.5:
                return self._invalidate_bbox()     
            else:
                return box
        except (AttributeError, NotImplementedError):
            return box
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _apply_transforms(self, img, transform):
        if isinstance(transform, NoOpTransform):
            return img
        else:
            return TransformList([transform]).apply_image(img)

    def _compare_lpips(self, box):    
        cropped0 = crop_and_pad(self.image, box)
        cropped1 = crop_and_pad(self.original, box)
        self._output("tools/lpips_augmented", cropped0)
        self._output("tools/lpips_original", cropped1)
        img0 = lpips.im2tensor(np.array(cropped0)).to(self.device)
        img1 = lpips.im2tensor(np.array(cropped1)).to(self.device)
        dist0 = self.loss_fn.forward(img0, img1)
        print(f'LPIPS for filename: {self.file_name.split("/")[-1]}: {dist0.item()}')
        return dist0.item()
    
    def _compare_psnr(self, box):    
        cropped0 = crop_and_pad(self.image, box)
        cropped1 = crop_and_pad(self.original, box)
        dist = PSNR(np.array(cropped0), np.array(cropped1))
        print(f'PSNR for filename: {self.file_name.split("/")[-1]}: {dist}')
        return dist
    
    def _compare_ssim(self, box):
        cropped0 = np.array(crop_and_pad(self.image, box))
        cropped1 = np.array(crop_and_pad(self.original, box))
        cropped0 = cv2.cvtColor(cropped0, cv2.COLOR_RGB2GRAY)
        cropped1 = cv2.cvtColor(cropped1, cv2.COLOR_RGB2GRAY)
        dist = ssim(cropped0, cropped1)
        print(f'SSIM for filename: {self.file_name.split("/")[-1]}: {dist}')
        return dist
    
    def _output(self, path, img):
        filepath = os.path.join(path, f'{self.file_name.split("/")[-1]}_{self.count}.jpg')
        self.count = self.count + 1
        print("Saving to {} ...".format(filepath))
        img.save(filepath)

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
        self.previous = TransformList(transforms) # previous transforms
        self.transformed = self.previous.apply_image(utils.read_image(self.file_name, format="BGR"))

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


# trained with classificator on images in bounding boxes
class SimpleBBTransform(Transform):

    def __init__(self, image: np.ndarray, file_name: str, transforms: list):
        super().__init__()
        self.image = image # transformed image
        self.file_name = file_name
        self.transforms = TransformList(transforms) # previous transforms
        self.original = Image.fromarray(image)
        self.transformed = self.transforms.apply_image(utils.read_image(self.file_name))
        #self.transformed = utils.convert_image_to_rgb(self.transformed, "BGR")
        #Image.fromarray(self.transformed).save(f'tools/transformed/{self.file_name.split("/")[-1]}_transformed.jpg')
        #self.original.save(f'tools/transformed/{self.file_name.split("/")[-1]}_original.jpg')
        self.device = f'cuda:{comm.get_rank()}'
        self.model = self._load_model()
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.count = 0

    def apply_image(self, img: np.ndarray):
        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            if self._predict(self.image, box) < 0.4:
                return self._invalidate_bbox()     
            else:
                return box
        except (AttributeError, NotImplementedError):
            return box
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _output(self, img):
        filename = str(uuid.uuid4())
        filepath = os.path.join("tools/test_vehicles", f'{self.file_name.split("/")[-1]}_{self.count}.jpg')
        #filepath = os.path.join("tools/vehicles", f'{filename}.jpg')
        self.count = self.count + 1
        print("Saving to {} ...".format(filepath))
        img.save(filepath)

    def _load_model(self):
        model = SimpleClassifier().to(self.device) # shift to GPU
        model.load_state_dict(torch.load('randaug/data/classifier/vehicle_classifier.pth'))
        model.eval()
        return model
    
    def _predict(self, image, box):    
        cropped = crop_and_pad(image, box)
        self._output(cropped)
        cropped = self.transforms(cropped)
        cropped = cropped.unsqueeze(0).to(0) # type: ignore
        return torch.sigmoid(self.model(cropped))

    def _invalidate_bbox(self):
        return np.array([np.Infinity,
                         np.Infinity,
                         np.Infinity,
                         np.Infinity])
    

def crop_and_pad(image: np.ndarray, box):

    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    im = Image.fromarray(image)
    im = im.crop(box[0])
    im = expand2square(im, (0, 0, 0))
    im = im.resize((224, 224))
    return im