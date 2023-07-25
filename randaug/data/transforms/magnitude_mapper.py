from detectron2.data.transforms.augmentation_impl import *


def map_magnitude_to_args(transform, magnitude: int):

    if transform == 'FixedSizeCrop':
        return {'crop_size': (256, 256)}
    elif transform == 'RandomBrightness':
        return {'intensity_min': 0, 'intensity_max': 10}
    elif transform == 'RandomFlip':
        return {'prob': 0.7}
    else:
        return 'Not Implemented'
    
