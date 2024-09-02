import clip
import faiss
import numpy as np
import torch
from PIL import Image

# pip install git+https://github.com/openai/CLIP.git


def load_model(device="cuda"):
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess


def embed_image(image, model, preprocess, device="cuda"):
    image = torch.unsqueeze(preprocess(image), 0).to(device)
    with torch.no_grad():
        return model.encode_image(image)


def index_data(data, index, distance="cosine"):
    data_cpu = data.cpu().numpy().astype(np.float32)
    if distance == "cosine":
        faiss.normalize_L2(data_cpu)
    index.add(data_cpu)
    return index


def search_image(img_path, index, model, preprocess, nres=5):
    query_img = Image.open(img_path)
    query = embed_image(query_img, model, preprocess)
    query = faiss.normalize_L2(query)
    D, I = index.search(np.array(query.cpu()).astype(np.float32), nres)
    return D, I
