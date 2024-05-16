import torch
import clip
from PIL import Image
import numpy as np
import torch.nn as nn

# from PIL import Image
# import requests
# from transformers import AutoProcessor, CLIPModel

# class CLIPClassifier(nn.Module):              
    
#     # device loading may be added here
#     def __init__(self, device):
#         super().__init__()
#         self.device = device
#         self.model, self.processor = self._init_clip_model()

#     def _init_clip_model(self):
#         model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
#         processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", device=self.device)

#         return model, processor

#     def forward(self, x):
#         inputs = self.processor(
#             text=["a photo without a vehicle", "a photo with a vehicle"], images=x, return_tensors="pt", padding=True
#         )

#         outputs = self.model(**inputs)

#         with torch.no_grad():
#             logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#             probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
#         return probs[0][1] # probability of being a vehicle

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
        print('clip')
        x = self.preprocess(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_logits, _ = self.model(x, self.preprocessed_text)
            proba_list = image_logits.softmax(dim=-1).cpu().numpy()[0]

        return proba_list[1] # probability of being a vehicle