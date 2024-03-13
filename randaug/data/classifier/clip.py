import torch
import clip
from PIL import Image
import numpy as np
import torch.nn as nn


class CLIPClassifier(nn.Module):              
    
    # device loading may be added here
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess = self._init_clip_model()
        self.preprocessed_text = clip.tokenize(["a photo without a vehicle", "a photo with a vehicle"]).to(self.device)          

    def _init_clip_model(self):
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def forward(self, x):
        x = self.preprocess(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_logits, _ = self.model(x, self.preprocessed_text)
            proba_list = image_logits.softmax(dim=-1).cpu().numpy()[0]

        return proba_list[1] # probability of being a vehicle