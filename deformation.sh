#!/bin/bash

# cyclegan
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cyclegan/fog/ --augmentation cycleganfog 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cyclegan/rain/ --augmentation cycleganrain 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cyclegan/snow/ --augmentation cyclegansnow 

# cut
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cut/fog/ --augmentation cutfog  
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cut/rain/ --augmentation cutrain  
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cut/snow/ --augmentation cutsnow  

# plugplay
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/plugplay_diffusion/0/fog/ --augmentation plugplayfog 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/plugplay_diffusion/0/rain/ --augmentation plugplayrain 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/plugplay_diffusion/0/snow/ --augmentation plugplaysnow  

# stablediffusion
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/stable_diffusion/0/fog/ --augmentation stablediffusionfog  
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/stable_diffusion/0/rain/ --augmentation stablediffusionrain 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/stable_diffusion/0/snow/ --augmentation stablediffusionsnow 

# cyclediffusion
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cycle_diffusion/0/fog/ --augmentation cyclediffusionfog 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cycle_diffusion/0/rain/ --augmentation cyclediffusionrain  
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/cycle_diffusion/0/snow/ --augmentation cyclediffusionsnow 

# controlnet
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/control_diffusion/0/fog/ --augmentation controlnetfog 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/control_diffusion/0/rain/ --augmentation controlnetrain 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/noboxes/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/control_diffusion/0/snow/ --augmentation controlnetsnow 

# mgie
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/crop/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/mgiediffusion/0/fog/ --augmentation mgiefog 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/crop/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/mgiediffusion/0/rain/ --augmentation mgierain 
python tools/analyse_deformation.py -d0 /mnt/ssd2/dataset/reference/crop/ -d1 /mnt/ssd2/dataset/cvpr24/adverse/itsc_augmentation/mgiediffusion/0/snow/ --augmentation mgiesnow 
