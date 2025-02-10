import numpy as np
import torch
import torchvision.transforms as T
import joblib
import torch.nn as nn

class DINOClassifier(nn.Module):

    # device loading may be added here
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.dino, self.model = self._init_dino_model()
        
    def _init_dino_model(self):
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        model = joblib.load('checkpoints/DINO_model_vehicle_classification_small_probability.pkl')
        return dino, model
    
    def forward(self, x):
        transform_image = T.Compose([T.ToTensor(), T.Resize(224), T.Normalize([0.5], [0.5])])
        x = transform_image(x)[:3].unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.dino(x)
            #prediction = self.model.predict(np.array(embedding[0].cpu()).reshape(1, -1))
            probabilities = self.model.predict_proba(np.array(embedding[0].cpu()).reshape(1, -1))
            #print(prediction)
            probabilities = probabilities.flatten()
            #print(probabilities)
        return probabilities[1] # probability of being a vehicle