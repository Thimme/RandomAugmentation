import numpy as np
from detectron2.data.transforms import Augmentation
from fvcore.transforms.transform import Transform, NoOpTransform
import albumentations as A

# from: https://github.com/facebookresearch/detectron2/issues/3054

# class AlbumentationsTransform(Transform):
#     def __init__(self, aug, size=()):
#         self.aug = aug
#         self.size = size
#         self.updated_boxes = []

#     def apply_coords(self, coords: np.ndarray) -> np.ndarray:
#         return coords

#     def apply_image(self, image):
#         return self.aug(image=image)["image"]

#     def apply_box(self, box: np.ndarray) -> np.ndarray:
#         try:
#             h, w = self.size[0], self.size[1]
#             normalized_box = box / [w, h, w, h] # normalize bounding box
#             transformed = np.array(self.aug(boxes=normalized_box.tolist())["bboxes"])
#             return transformed * [w, h, w, h]
#         except (AttributeError, NotImplementedError):
#             return box

#     def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
#         try:
#             return self.aug(image=segmentation)["image"]
#         except AttributeError:
#             return segmentation
        

class AlbumentationsTransform(Transform):
    def __init__(self, augmentor, size=()):
        self.augmentor = A.ReplayCompose([augmentor])  # This is a ReplayCompose
        self.size = size
        self.replay = None  # To store replay params

    def apply_image(self, image):
        result = self.augmentor(image=image)
        self.replay = result["replay"]  # Save the replay after applying to the image
        return result["image"]

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        if self.replay is None:
            return box  # apply_image() should be called first

        h, w = self.size
        normalized_box = box / [w, h, w, h]
        replay_result = A.ReplayCompose.replay(
            self.replay,
            bboxes=normalized_box
        )

        transformed_bboxes = replay_result["bboxes"]
        #print(transformed_bboxes)
        if transformed_bboxes is None or np.array(transformed_bboxes).size == 0:
            return self._invalidate_bbox()
        return transformed_bboxes * [w, h, w, h]


    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if self.replay is None:
            return segmentation  # apply_image() should be called first

        replay_result = A.ReplayCompose.replay(self.replay, image=segmentation)
        return replay_result["image"]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def _invalidate_bbox(self):
        return np.array([np.Infinity,
                         np.Infinity,
                         np.Infinity,
                         np.Infinity])


# class AlbumentationsWrapper(Augmentation):
#     """
#     Updated wrapper for Albumentations transforms in Detectron2.
#     Compatible with latest Albumentations versions.
#     """

#     def __init__(self, augmentor):
#         self._aug = augmentor

#     def get_transform(self, image):
#         do = self._rand_range() < self._aug.p
#         if do:
#             return AlbumentationsTransform(self._aug, size=image.shape[:2])
#         else:
#             return NoOpTransform()
        


# def prepare_param(aug, image):
#         params = aug.get_params()
#         if aug.targets_as_params:
#             targets_as_params = {"image": image}
#             params_dependent_on_targets = aug.get_params_dependent_on_targets(targets_as_params)
#             params.update(params_dependent_on_targets)
#         params = aug.update_params(params, **{"image": image})
#         return params


# class AlbumentationsWrapper(Augmentation):
#     """
#     Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
#     Image, Bounding Box and Segmentation are supported.
#     Example:
#     .. code-block:: python
#         import albumentations as A
#         from detectron2.data import transforms as T
#         from detectron2.data.transforms.albumentations import AlbumentationsWrapper

#         augs = T.AugmentationList([
#             AlbumentationsWrapper(A.RandomCrop(width=256, height=256)),
#             AlbumentationsWrapper(A.HorizontalFlip(p=1)),
#             AlbumentationsWrapper(A.RandomBrightnessContrast(p=1)),
#         ])  # type: T.Augmentation

#         # Transform XYXY_ABS -> XYXY_REL
#         h, w, _ = IMAGE.shape
#         bbox = np.array(BBOX_XYXY) / [w, h, w, h]

#         # Define the augmentation input ("image" required, others optional):
#         input = T.AugInput(IMAGE, boxes=bbox, sem_seg=IMAGE_MASK)

#         # Apply the augmentation:
#         transform = augs(input)
#         image_transformed = input.image  # new image
#         sem_seg_transformed = input.sem_seg  # new semantic segmentation
#         bbox_transformed = input.boxes   # new bounding boxes

#         # Transform XYXY_REL -> XYXY_ABS
#         h, w, _ = image_transformed.shape
#         bbox_transformed = bbox_transformed * [w, h, w, h]
#     """

#     def __init__(self, augmentor):
#         """
#         Args:
#             augmentor (albumentations.BasicTransform):
#         """
#         # super(Albumentations, self).__init__() - using python > 3.7 no need to call rng
#         self._aug = augmentor

#     def get_transform(self, image):
#         do = self._rand_range() < self._aug.p
#         if do:
#             params = prepare_param(self._aug, image)
#             return AlbumentationsTransform(self._aug, params)
#         else:
#             return NoOpTransform()
        