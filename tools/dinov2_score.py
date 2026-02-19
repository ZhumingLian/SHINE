from transformers import AutoImageProcessor, AutoModel
import torch.nn as nn
import torch

def init_dinov2_model(dinov2_model, device):
    processor = AutoImageProcessor.from_pretrained(dinov2_model)
    model = AutoModel.from_pretrained(dinov2_model)
    model.eval()
    model.to(device)
    return model, processor


def calculate_dinov2_score(model, processor, image1, image2, device):
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        image_features1 = outputs1.last_hidden_state
        image_features1 = image_features1.mean(dim=1)
            
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        outputs2 = model(**inputs2)
        image_features2 = outputs2.last_hidden_state
        image_features2 = image_features2.mean(dim=1)
        
        cos = nn.CosineSimilarity(dim=0)
        sim = cos(image_features1[0],image_features2[0]).item()
        sim = (sim+1)/2
        # print('Similarity:', sim)
        return sim
