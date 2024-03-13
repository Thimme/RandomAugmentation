import numpy as np
import lpips
import torch
import torchvision.transforms
import cv2
import uuid
import os
from detectron2.utils import comm
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
from math import log10, sqrt 
from fvcore.transforms.transform import Transform, TransformList, NoOpTransform
from skimage.metrics import structural_similarity as ssim
from randaug.data.classifier.classifier import SimpleClassifier
from randaug.data.classifier.clip import CLIPClassifier
from randaug.data.classifier.dino import DINOClassifier
from detectron2.data import detection_utils as utils


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
            if self._compare_lpips(box=box) > 0.5:
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
        cropped0 = crop_and_pad(self.image, box, resize=True)
        cropped1 = crop_and_pad(self.original, box, resize=True)
        img0 = lpips.im2tensor(np.array(cropped0)).to(self.device)
        img1 = lpips.im2tensor(np.array(cropped1)).to(self.device)
        dist = self.loss_fn.forward(img0, img1).item()
        self._output("tools/lpips224", cropped0, cropped1, dist)
        return dist
    
    def _compare_psnr(self, box):    
        cropped0 = crop_and_pad(self.image, box)
        cropped1 = crop_and_pad(self.original, box)
        dist = PSNR(np.array(cropped0), np.array(cropped1))
        self._output("tools/ssim", cropped0, cropped1, dist)
        return dist
    
    def _compare_ssim(self, box):
        cropped0 = crop_and_pad(self.image, box)
        cropped1 = crop_and_pad(self.original, box)
        cropped0 = cv2.cvtColor(np.array(cropped0), cv2.COLOR_RGB2GRAY)
        cropped1 = cv2.cvtColor(np.array(cropped1), cv2.COLOR_RGB2GRAY)
        dist = ssim(cropped0, cropped1)
        self._output("tools/ssim", cropped0, cropped1, dist)
        return dist
    
    def _output(self, path, im1, im2, lpips):
        filepath = os.path.join(path, f'{self.file_name.split("/")[-1]}_{self.count}.jpg')
        self.count = self.count + 1
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        draw = ImageDraw.Draw(dst)
        draw.text((0, 0),f"{lpips:.2f}",(255,255,255))
        dst.save(filepath)

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
class CLIPBBTransform(Transform):

    def __init__(self, image: np.ndarray, file_name: str, transforms: list):
        super().__init__()
        self.image = image # transformed image
        self.device = f'cuda:{comm.get_rank()}'
        self.model = CLIPClassifier(device=self.device)#.to(self.device)
        self.threshold = 0.5

    def apply_image(self, img: np.ndarray):
        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            if self._predict(self.image, box) < self.threshold:
                #print('removeu')
                return self._invalidate_bbox()     
            else:
                #print('manteve')
                return box
        except (AttributeError, NotImplementedError):
            return box
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _predict(self, image, box):    
        cropped = crop_and_pad(image, box)
        #cropped = cropped.unsqueeze(0).to(self.device) # type: ignore
        return self.model(cropped)

    def _invalidate_bbox(self):
        return np.array([np.Infinity,
                         np.Infinity,
                         np.Infinity,
                         np.Infinity])


# trained with classificator on images in bounding boxes
class DINOBBTransform(Transform):

    def __init__(self, image: np.ndarray, file_name: str, transforms: list):
        super().__init__()
        self.image = image # transformed image
        self.device = f'cuda:{comm.get_rank()}'
        self.model = DINOClassifier(device = self.device)#.to(self.device)

    def apply_image(self, img: np.ndarray):
        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            pred = self._predict(self.image, box)
            if pred == 'non-vehicles':
                #print('removeu')
                return self._invalidate_bbox()     
            else:
                #print('manteve')
                return box
        except (AttributeError, NotImplementedError):
            return box
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _predict(self, image, box):    
        cropped = crop_and_pad(image, box)
        #cropped = cropped.unsqueeze(0).to(self.device) # type: ignore
        return self.model(cropped)

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

    def _load_model(self):
        model = SimpleClassifier().to(self.device) # shift to GPU
        model.load_state_dict(torch.load('randaug/data/classifier/vehicle_classifier.pth'))
        model.eval()
        return model
    
    def _predict(self, image, box):    
        #cropped = crop_and_pad(image, box)
        cropped = crop(image, box)
        cropped = self.transforms(cropped)
        cropped = cropped.unsqueeze(0).to(0) # type: ignore
        return torch.sigmoid(self.model(cropped))

    def _invalidate_bbox(self):
        return np.array([np.Infinity,
                         np.Infinity,
                         np.Infinity,
                         np.Infinity])
    

# trained with classificator on images in bounding boxes
class OutputBBTransform(Transform):

    def __init__(self, image: np.ndarray, file_name: str, transforms: list):
        super().__init__()
        self.image = image # transformed image
        self.file_name = file_name
        self.transforms = TransformList(transforms) # previous transforms
        self.original = np.array(Image.open(self.file_name))
        self.transformed = self.transforms.apply_image(utils.read_image(self.file_name))
        self.count = 0

    def apply_image(self, img: np.ndarray):
        return img
    
    def apply_box(self, box: np.ndarray) -> np.ndarray:
        cropped = crop_and_pad(self.image, box)
        cropped_original = crop_and_pad(self.original, box)
        filename = str(uuid.uuid4())
        self._output(cropped_original, filename, "tools/out/vehicles_original/")
        self._output(cropped, filename, "tools/out/vehicles_augmented/")
        return box
            
    def apply_coords(self, coords):
        return coords
    
    def apply_segmentation(self, segmentation):
        return segmentation
    
    def _output(self, img, filename, path):
        filepath = os.path.join(path, f'{filename}.jpg')
        self.count = self.count + 1
        print("Saving to {} ...".format(filepath))
        img.save(filepath)


def crop_and_pad(image: np.ndarray, box, resize=True):

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
    if resize:
        im = im.resize((224, 224))
    return im


def crop(image: np.ndarray, box):
    im = Image.fromarray(image)
    im = im.crop(box[0])
    #im = im.resize((224, 224))
    return im
