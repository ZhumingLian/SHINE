import torch
import clip
from PIL import Image


class SimilarityCalculator:
    def __init__(self, clip_model, device):
        self.device = device
        self.model, self.preprocess = self._initialize_model(clip_model, self.device)
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=0)
        self.image_1 = None
        self.image_2 = None
        self.raw_similarity = None

    def _initialize_model(self, model_name="ViT-B/32", device="cpu"):
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess

    def _embed_image(self, image):
        preprocessed_image = (
            self.preprocess(image).unsqueeze(0).to(self.device)
        )
        image_embeddings = self.model.encode_image(preprocessed_image)
        return image_embeddings

    def calculate_similarity(self, image_1, image_2):
        self.image_1 = self._embed_image(image_1)
        self.image_2 = self._embed_image(image_2)
        self.raw_similarity = self.cosine_similarity(
            self.image_1[0], self.image_2[0]
        ).item()
        return self.raw_similarity