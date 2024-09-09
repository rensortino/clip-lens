import torch
from transformers import AutoImageProcessor, AutoModel

class EmbeddingModel:
    def __init__(self, model_id, device="cuda"):
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.preprocessor = AutoImageProcessor.from_pretrained(model_id)
        self.device = device
    
    def encode_image(self, x):
        with torch.no_grad():
            sample = self.preprocessor(x, return_tensors="pt").to(self.device)
            if "dino" in str(self.model.__class__).lower():
                # In DINOv2, pooler_output == last_hidden_state[:,0], while in ViT the weights of ViTPooler are not initialized, 
                # so we need to explicitly take the class token
                return self.model(**sample).pooler_output 
            elif "clip" in str(self.model.__class__).lower():
                return self.model.get_image_features(**sample)
            elif "vit" in str(self.model.__class__).lower():
                return self.model(**sample).last_hidden_state[:, 0]
            else:
                raise ValueError(f"Model {self.model} not supported")