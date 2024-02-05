import torch
import os
import numpy as np
import kornia
import math
import torch.nn as nn
import random
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from math import floor, ceil
import gc


class DropModel:
    def __init__(self,
                 imsize = (256, 256),
                 noise_resize = 25,
                 size_threshold = 0.015,
                 frequency_threshold = 8,
                 shape_threshold = 0.7,
                 min_thickness = 0.3,
                 max_thickness = 0.8,
                 sigma_size = 3, 
                 device = None
                 ):

        self.device = device

        # higher = more frequent
        self.frequency_threshold = frequency_threshold

        # smaller = bigger drops
        self.size_threshold = size_threshold

        # higher = more similar to a circle
        self.shape_threshold = shape_threshold

        # thickness
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

        # higher = drops angles are more continuous
        self.noise_resize = noise_resize

        self.imsize = imsize
        #self.r = torch.tensor([2, 1]).to(self.device).unsqueeze(-1)
        self.r = torch.tensor([2, 1]).to(self.device).unsqueeze(-1)

        #uv = torch.tensor(np.indices(self.imsize)).permute(1, 2, 0).float().to(self.device)
        uv = torch.tensor(np.indices(self.imsize)).permute(1, 2, 0).float().to(self.device)
        uv[:, :, 0] /= self.imsize[0]
        uv[:, :, 1] /= self.imsize[1]
        uv = uv.view(-1, 2)
        self.uv = uv
        #self.uv = self.uv.repeat(self.r.size(0), 1, 1)
        self.noise_h = int(self.imsize[0] / self.noise_resize)
        self.noise_w = int(self.imsize[1] / self.noise_resize)

        self.blur = kornia.filters.GaussianBlur2d((11, 11), (sigma_size, sigma_size))
        self.maskblur = kornia.filters.GaussianBlur2d((11, 11), (sigma_size, sigma_size))

        #x = (torch.tensor(self.imsize).to(self.device) * self.r * self.size_threshold).unsqueeze(1)
        x = (torch.tensor(self.imsize).to(self.device) * self.r * self.size_threshold).unsqueeze(1)

        numsquares = (torch.round(self.uv * x - 0.5) % x + 1).max(dim = 1)[0]
        numsquares = numsquares[:, 0] * numsquares[:, 1]
        # The probability to become a drop center depends on the size
        # The formula is expressed like this because of the original implementation
        ratio_squares = numsquares / (self.imsize[0] * self.imsize[1])
        self.probabilities = 1 - (self.frequency_threshold - self.r.squeeze()) * .08 * ratio_squares

        # generating single drop
        self.drop_size = 256
        self.noise_resize = int(self.drop_size / 2)
        #single_drop = torch.tensor(np.indices((self.drop_size, self.drop_size))).permute(1, 2, 0).float().to(self.device)
        single_drop = torch.tensor(np.indices((self.drop_size, self.drop_size))).permute(1, 2, 0).float().to(self.device)

        single_drop[:, :, 0] /= self.drop_size
        single_drop[:, :, 1] /= self.drop_size

        self.single_drop = single_drop

        bias = np.zeros((imsize[0], imsize[1], 1), np.uint8) * 255
        top = (int(imsize[1] / 2), int(imsize[0] / 10 * 5.5))
        bottom_1 = (0, imsize[0])
        bottom_2 = (imsize[1], imsize[0])
        triangle_cnt = np.array([top, bottom_1, bottom_2])
        cv2.drawContours(bias, [triangle_cnt], 0, 255, -1)
        #bias = (1 - torch.tensor(bias).permute(2, 0, 1).to(self.device).float() / 255)
        bias = (1 - torch.tensor(bias).permute(2, 0, 1).to(self.device).float() / 255)
        bias = bias.unsqueeze(0)
        self.bias = self.maskblur(bias)


    def get_drop_stock(self, numdrops):

        #print(f"numdrops: {numdrops} - device: {self.device}")
        single_drop = self.single_drop.repeat(numdrops, 1, 1)
        single_drop = single_drop.view(numdrops, -1, 2)

        #noise = torch.rand(dtype=torch.float16, numdrops, 3, int(self.drop_size / self.noise_resize), int(self.drop_size / self.noise_resize)).to(self.device)
        noise = torch.rand(numdrops, 3, int(self.drop_size / self.noise_resize), int(self.drop_size / self.noise_resize), dtype=torch.float16).to(self.device)      
        noise = F.interpolate(noise, size = (self.drop_size, self.drop_size), mode = 'bilinear')
        noise = noise.view(numdrops, 3, -1).permute(0, 2, 1)
        n = noise[:, :, 2].unsqueeze(-1)

        p = math.pi * 2 * (single_drop - 0.25) + (n - 0.5) * 2
        s = torch.sin(p)
        t = (s[:, :, 0] + s[:, :, 1])

        k = torch.cos(p)
        interp = (0.2 * (1 - t - 0.5) + 2. * (t - 0.5)).unsqueeze(-1)
        v = - torch.cat((k, interp), -1)

        # normalization
        v /= torch.sqrt(v[:, :,  0] ** 2 + v[:, :, 1] ** 2 + v[:, :, 2] ** 2).unsqueeze(-1)
        v = v.permute(0, 2, 1)
        v[:, 2, :] = random.uniform(self.min_thickness, self.max_thickness)
        v = v.view(numdrops, 3, self.drop_size, self.drop_size)
        mask = (t > self.shape_threshold).to(dtype=torch.float16)
        mask = mask.view(numdrops, 1, self.drop_size, self.drop_size)

        return torch.cat((v * mask, mask), dim = 1)


    def get_numdrops(self):

        #random_map = torch.rand(self.r.size(0), self.imsize[0], self.imsize[1]).to(self.device)
        random_map = torch.rand(self.r.size(0), self.imsize[0], self.imsize[1]).to(self.device)
        probs = self.probabilities.unsqueeze(-1).unsqueeze(-1)

        numdrops = random_map.gt(probs).float().sum(dim = 1).sum(dim = 1)
        while numdrops.eq(0).sum() > 0:
            #random_map = torch.rand(self.r.size(0), self.imsize[0], self.imsize[1]).to(self.device)
            random_map = torch.rand(self.r.size(0), self.imsize[0], self.imsize[1]).to(self.device)

            probs = self.probabilities.unsqueeze(-1).unsqueeze(-1)

            numdrops = random_map.gt(probs).float().sum(dim=1).sum(dim=1)
        return numdrops.long()

    def get_drops_sizes(self):

        #x = (torch.tensor(self.imsize).to(self.device) * self.r * self.size_threshold).unsqueeze(1)
        x = (torch.tensor(self.imsize).to(self.device) * self.r * self.size_threshold).unsqueeze(1)

        numsquares = (torch.round(self.uv * x - 0.5) % x + 1).max(dim = 1)[0]

        #sizes = (torch.tensor(self.imsize).to(self.device) / numsquares).min(dim = 1)[0].long()
        sizes = (torch.tensor(self.imsize).to(self.device) / numsquares).min(dim = 1)[0].long()

        return sizes
    

    def blend(self, t):
        # average over dim zero but only on nonzero coordinates nonzero
        nonzero_mask = (t != 0)
        nonzero_count = nonzero_mask.sum(dim = 0)
        nonzero_count[nonzero_count == 0] = 1
        return t.sum(dim = 0) / nonzero_count

    
    def split_list(self, alist, wanted_parts=1):
        length = len(alist)
        return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
                for i in range(wanted_parts) ]
    

    def get_normal_map(self):
        numdrops = self.get_numdrops() / 1.5 # reduce drops to reduce memory overhead on gpus (1.7 - 1.8)
        numdrops_total = int(numdrops.sum().item())
        drops_sizes = self.get_drops_sizes()
        #drops_stock = self.get_drop_stock(numdrops_total).to(dtype=torch.float16)

        rand_y = torch.randint(- int(self.imsize[0] / 2), int(self.imsize[0] / 2), size = (numdrops_total, 1)).to(self.device)
        rand_x = torch.randint(- int(self.imsize[1] / 2), int(self.imsize[1] / 2) , size = (numdrops_total, 1)).to(self.device)
        offsets = torch.cat((rand_x, rand_y), dim = 1).to(dtype=torch.float16)

        dropmap = torch.zeros(1, 3, self.imsize[0], self.imsize[1], dtype=torch.float16).to(self.device)
        maskmap = torch.zeros(1, 1, self.imsize[0], self.imsize[1], dtype=torch.float16).to(self.device)

        drops = torch.zeros(1, 3, self.imsize[0], self.imsize[1], dtype=torch.float16).to(self.device)
        masks = torch.zeros(1, 1, self.imsize[0], self.imsize[1], dtype=torch.float16).to(self.device)

        for layer_id in range(0, self.r.size(0)):
            # Get the drops and resize them to the desired size
            startindex = numdrops[:layer_id].sum().long().item()
            endindex = numdrops[:layer_id + 1].sum().long().item()
            layer_size = drops_sizes[layer_id].long().item()
            # drops_layer = drops_stock[startindex:endindex]
            
            # Padding
            pad_value_left = floor((self.imsize[0] - layer_size) / 2)
            pad_value_right = ceil((self.imsize[0] - layer_size) / 2)
            pad_value_top = floor((self.imsize[1] - layer_size) / 2)
            pad_value_bottom = ceil((self.imsize[1] - layer_size) / 2)
            
            # Process in smaller batches
            batch_size = 25  # Adjust based on GPU memory
            for start in range(startindex, endindex, batch_size): # type: ignore
                end = min(start + batch_size, endindex)
                batch_size = end - start
                drops_batch = self.get_drop_stock(batch_size).to(dtype=torch.float16)

                # Apply transformations
                drops_batch = F.interpolate(drops_batch, size=(layer_size, layer_size), mode='bilinear')
                drops_batch = F.pad(drops_batch, (pad_value_top, pad_value_bottom, pad_value_left, pad_value_right), value=0)
                drops_batch = kornia.geometry.transform.translate(drops_batch, offsets[start:end])

                masks_batch = drops_batch[:, 3].unsqueeze(1)
                drops_batch = drops_batch[:, :3]

                # Getting a single displacement map
                drops_concat = torch.cat((drops, drops_batch), 0)
                masks_concat = torch.cat((masks, masks_batch), 0)

                drops = self.blend(drops_concat).unsqueeze(0)
                masks = self.blend(masks_concat).unsqueeze(0)

                del masks_batch, drops_batch
                torch.cuda.empty_cache()
        
        dropmap[dropmap == 0] = drops[dropmap == 0]
        maskmap[maskmap == 0] = masks[maskmap == 0]

        return dropmap.permute(0, 2, 3, 1).view(1, self.imsize[0] * self.imsize[1], 3), maskmap.permute(0, 2, 3, 1).view(1, self.imsize[0] * self.imsize[1], 1)


    def get_noise(self):
        #noise = torch.rand(self.r.size(0), 3, self.noise_h, self.noise_w).to(self.device)
        noise = torch.rand(self.r.size(0), 3, self.noise_h, self.noise_w).to(self.device)

        noise = F.interpolate(noise, size = self.imsize, mode='bilinear')

        return noise.view(self.r.size(0), 3, -1).permute(0, 2, 1)

    def add_drops(self, im, sigma, with_bias = False,  return_drops = False):
        #sigma = torch.zeros(1).fill_(30).to(self.device)
        #transparency = torch.zeros(1).fill_(1).to(self.device)
        transparency = torch.zeros(1).fill_(1).to(self.device)

        v, mask = self.get_normal_map()
        # Refraction
        im_coords = (self.uv * 2 - 1)
        refraction_coords = - v[:, :, :2] * v[:, :, 2].unsqueeze(-1)

        buffer = (im_coords + refraction_coords).view(1, self.imsize[0], self.imsize[1], 2)

        # requires shifting
        b_buf = buffer[:, :, :, 0].clone()
        buffer[:, :, :, 0] = buffer[:, :, :, 1]
        buffer[:, :, :, 1] = b_buf

        im_refracted = F.grid_sample(
            im,
            buffer, 
            mode = 'bilinear', 
            padding_mode='reflection', 
            align_corners=True)

        #blur_kernel = get_gaussian_kernel(sigma=sigma).to(self.device)
        blur_kernel = self.get_gaussian_kernel(sigma=sigma).to(self.device)
        im_refracted = F.conv2d(im_refracted, blur_kernel, groups=3, padding=int((blur_kernel.size(2) - 1) / 2))

        # We randomly select some drops
        #mask = ((t > self.shape_threshold).view(layers, 1, self.imsize[0], self.imsize[1]))

        # mask = ((t > self.shape_threshold)).view(layers, 1, self.imsize[0], self.imsize[1])
        maskblur_kernel = self.get_gaussian_kernel(sigma=sigma, channels=1)

        mask = mask.view(1, 1, self.imsize[0], self.imsize[1])
        mask = F.conv2d(mask.float(), maskblur_kernel, padding=int((blur_kernel.size(2) - 1) / 2))
        # adding to the image from the smallest to the biggest

        if with_bias:
            mask = mask * self.bias
        
        im = (im_refracted * mask * transparency) + im * (1 - (mask * transparency))
        
        if return_drops:
            mask_binary = torch.zeros(mask.size()).to(self.device)
            mask_binary[mask > 0] = 1
            return im, mask_binary
        
        return im
    

    def get_gaussian_kernel(self, sigma=torch.ones(1), channels=3):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        sigma_detach = sigma.clone().detach().item() * 2
        sigma_detach = math.ceil(sigma_detach)
        if sigma_detach % 2 == 0:
            kernel_size = sigma_detach + 1
        else:
            kernel_size = sigma_detach

        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        #xy_grid = torch.stack([x_grid, y_grid], dim=-1).float().to(device)
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float().to(self.device)

        mean = (kernel_size - 1) / 2.
        #variance = torch.pow(sigma, 2).to(self.device)
        variance = torch.pow(sigma, 2).to(self.device)

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                        torch.exp(
                            -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                            (2 * variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        return gaussian_kernel
