import numpy as np
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform

# from: https://github.com/facebookresearch/detectron2/issues/3054

class AlbumentationsTransform(Transform):
    def __init__(self, aug, params, size=()):
        self.aug = aug
        self.params = params
        self.size = size

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_image(self, image):
        return self.aug.apply(image, **self.params)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            h, w = self.size[0], self.size[1]
            normalized_box = box / [w, h, w, h] # normalize bounding box
            transformed = np.array(self.aug.apply_to_bboxes(normalized_box.tolist(), **self.params))
            return transformed * [w, h, w, h]
        except (AttributeError, NotImplementedError):
            return box

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_mask(segmentation, **self.params)
        except AttributeError:
            return segmentation


def prepare_param(aug, image):
        params = aug.get_params()
        if aug.targets_as_params:
            targets_as_params = {"image": image}
            params_dependent_on_targets = aug.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)
        params = aug.update_params(params, **{"image": image})
        return params


class AlbumentationsWrapper(Augmentation):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Image, Bounding Box and Segmentation are supported.
    Example:
    .. code-block:: python
        import albumentations as A
        from detectron2.data import transforms as T
        from detectron2.data.transforms.albumentations import AlbumentationsWrapper

        augs = T.AugmentationList([
            AlbumentationsWrapper(A.RandomCrop(width=256, height=256)),
            AlbumentationsWrapper(A.HorizontalFlip(p=1)),
            AlbumentationsWrapper(A.RandomBrightnessContrast(p=1)),
        ])  # type: T.Augmentation

        # Transform XYXY_ABS -> XYXY_REL
        h, w, _ = IMAGE.shape
        bbox = np.array(BBOX_XYXY) / [w, h, w, h]

        # Define the augmentation input ("image" required, others optional):
        input = T.AugInput(IMAGE, boxes=bbox, sem_seg=IMAGE_MASK)

        # Apply the augmentation:
        transform = augs(input)
        image_transformed = input.image  # new image
        sem_seg_transformed = input.sem_seg  # new semantic segmentation
        bbox_transformed = input.boxes   # new bounding boxes

        # Transform XYXY_REL -> XYXY_ABS
        h, w, _ = image_transformed.shape
        bbox_transformed = bbox_transformed * [w, h, w, h]
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        # super(Albumentations, self).__init__() - using python > 3.7 no need to call rng
        self._aug = augmentor

    def get_transform(self, image):
        do = self._rand_range() < self._aug.p
        if do:
            params = prepare_param(self._aug, image)
            return AlbumentationsTransform(self._aug, params)
        else:
            return NoOpTransform()