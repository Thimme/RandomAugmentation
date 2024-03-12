import numpy as np
import torch
import torchvision.transforms as T
import joblib
import torch.nn as nn

class DINOClassifier(nn.Module):

    # device loading may be added here
    def __init__(self, device):
        super().__init__()
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dinov2_vitg14, self.model = self._init_dino_model(device)
        
    def _init_dino_model(self, device):
        dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
        #dinov2_vitg14.to(device)
        model = joblib.load('/home/mayara/detection/RandomAugmentation/randaug/data/classifier/DINO_model_vehicle_classification.pkl') #bad code
        return dinov2_vitg14, model
    
    def forward(self, x):
        transform_image = T.Compose([T.ToTensor(), T.Resize(224), T.Normalize([0.5], [0.5])])
        x = transform_image(x)[:3].unsqueeze(0)

        with torch.no_grad():
            #embedding = self.dinov2_vitg14(x.to(self.device))
            embedding = self.dinov2_vitg14(x)
            prediction = self.model.predict(np.array(embedding[0].cpu()).reshape(1, -1))
        
        return prediction[0] #return 'vehicles' or 'non-vehicles'