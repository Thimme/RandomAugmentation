from torchvision import datasets, models, transforms
import torch.nn as nn
import clip
import torch
import numpy as np
import joblib
import torchvision.transforms as T
from PIL import Image


# retrain simple classifier with 
class SimpleClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.resnet(x)
        return x
    

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
    

class DINOClassifier(nn.Module):

    # device loading may be added here
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.dino, self.svm = self._init_dino_model()
        
    def _init_dino_model(self):
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        svm = joblib.load('checkpoints/dino.pkl')
        return dino, svm
    
    def forward(self, x):
        transform_image = T.Compose([T.ToTensor(), T.Resize(224), T.Normalize([0.5], [0.5])])
        x = transform_image(x)[:3].unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.dino(x)
            probabilities = self.svm.predict_proba(np.array(embedding[0].cpu()).reshape(1, -1)).flatten()

        return probabilities[1]