import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import normalize
from torch.utils.data import Dataset
from tqdm import tqdm
from umap import UMAP
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from model import EmbeddingModel


class ImageDataset(Dataset):
    def __init__(self, data_root, preprocessor):
        super().__init__()
        self.img_paths = np.array([p.as_posix() for p in Path(data_root).glob("**/*.png")])
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int) -> torch.Tuple[torch.Any]:
        path = self.img_paths[index]
        img = Image.open(path)
        return img, path, index

def visualize_embeddings(emb_dict):
    # Visualize embeddings
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        for i, name, embs in enumerate(emb_dict.items()):
            ax.scatter(embs[:, 0], embs[:, 1], embs[:, 2], c=mcolors[i % len(mcolors)], label=name)
            # ax.scatter(dino_reduced[:, 0], dino_reduced[:, 1], dino_reduced[:, 2], c='blue', label='DINOv2')

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.legend()
        plt.title('Image Embeddings')
        plt.show()
        plt.savefig("plot.png")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Compute Embeddings")
    parser.add_argument("--config", default="configs/embeddings/clip/imagenet.yaml")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--model-id", default="openai/clip-vit-base-patch16")
    parser.add_argument("--data-root", required=True)

    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EmbeddingModel(args.model_id, device)

    dataset = ImageDataset(args.data_root, model.preprocessor)

    emb_dir = Path("embeddings") / args.data_root / args.model_id.split("/")[-1]
    path_str_type = "S350"
    
    emb_memory_loc = (emb_dir / "embeddings.npy").as_posix()
    paths_memory_loc = (emb_dir / "paths.npy").as_posix()
    parameters_path = (emb_dir / "params.yaml").as_posix()
    
    emb_dir.mkdir(exist_ok=True, parents=True)

    # -- Get and store 1)encodings 2)path to each example
    print("Get encoding...")
    with torch.no_grad():
        embeddings = torch.tensor([], device=device)
        paths = []
        for image, path, index in tqdm(dataset):
            emb = model.encode_image(image)
            embeddings = torch.cat([embeddings, normalize(emb, dim=1)])
            paths.append(path)

    sim_heatmap = embeddings @ embeddings.T

    umap = UMAP(n_components=3, random_state=42)
    proj_umap = umap.fit_transform(embeddings.cpu())
    visualize_embeddings({"clip": proj_umap})
    
    
    pca = PCA(n_components=2)
    proj_pca = pca.fit_transform(embeddings)
