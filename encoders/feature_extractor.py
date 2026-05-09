import torch.nn as nn
import torch
from encoders.r2d2_encoder.export_image_embeddings import get_pretrained_model
import torchvision.transforms as tvf
from encoders.sp_encoder.export_image_embeddings import SuperPoint
from encoders.XFeat.modules.xfeat import XFeat

class FeatureExtractor(nn.Module):
    def __init__(self, feature_type):
        super(FeatureExtractor, self).__init__()
        self.feature_type = feature_type
        if feature_type == "sp":
            print("Loading SuperPoint model...")
            self.model = SuperPoint().cuda().eval()
            self.feature_dim = 256
        elif feature_type == "xfeat":
            print("Loading XFeat model...")
            self.model = XFeat()
            self.feature_dim = 64
        else:
            raise ValueError("Foundation model not supported")
        

    @torch.no_grad()
    def detectAndCompute(self, image, top_k = None, detection_threshold = None):
        if self.feature_type == "sp":
            return self.model.detectAndCompute(image, top_k, detection_threshold)

        elif self.feature_type == "xfeat":
            return self.model.detectAndCompute(image, top_k, detection_threshold)
        
    @torch.no_grad()
    def detectAndComputeDense(self, image):
        if self.feature_type == "sp":
            return self.model.detectAndComputeDense(image)
        elif self.feature_type == "xfeat":
            return self.model.get_descriptors_and_scores(image)

    @torch.no_grad()
    def forward(self, image):
        if self.feature_type == "sp":
            features, scores = self.model(image)
            return {
                "feature_map": features,
                "scores": scores
            }
        elif self.feature_type == "xfeat":
            pass
