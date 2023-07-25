import numpy as np
import torch
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform
from randaug.data.transforms.corruptions import corrupt


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
            

class MyColorAugmentation(T.Augmentation):

    def get_transform(self, image):
        return T.ColorTransform(self.color_func)
    
    def color_func(self, x):
        r = np.random.rand(2)
        return np.float32(x * r[0] + r[1] * 10)
        

class WeatherAugmentation(T.Augmentation):

    def __init__(self, name, severity=1):
        super().__init__()
        self.name = name
        self.severity = severity

    def get_transform(self, image):
        return WeatherTransform(name=self.name, severity=self.severity)
    

class RandomAugmentation():
    
    def __init__(self, cfg, transforms):
        self.N = cfg.rand_N
        self.M = cfg.rand_M
        self.transforms = transforms
    
    def __repr__(self):
        repr = '-'.join([t.__class__.__name__ for t in self.transforms])
        return f'{self.N}-{self.M}-{repr}'
    
    def get_transforms(self):
        return self.transforms


    