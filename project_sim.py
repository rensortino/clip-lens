import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from umap import UMAP

from model import EmbeddingModel
from projector import EmbeddingProjector


# Function to process images and compute embeddings
def process_images(folder_path, clip_model, dino_model):
    clip_embeddings = []
    dino_embeddings = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            clip_embedding = clip_model.encode_image(Image.open(image_path)).squeeze().cpu().numpy()
            dino_embedding = dino_model.encode_image(Image.open(image_path)).squeeze().cpu().numpy()
            clip_embeddings.append(clip_embedding)
            dino_embeddings.append(dino_embedding)
    return np.array(clip_embeddings), np.array(dino_embeddings)

# Main script
def main(args):

    clip_model = EmbeddingModel("openai/clip-vit-base-patch32")
    dino_model = EmbeddingModel("facebook/dinov2-base")

    # Compute embeddings
    clip_embeddings, dino_embeddings = process_images(args.img_dir, clip_model, dino_model)

    projector = EmbeddingProjector()
    projector.save_embeddings(clip_embeddings)
    projector.save_embeddings(dino_embeddings)
    projector.generate_sprite_images(args.img_dir)
    projector.generate_sprite_images(args.img_dir)

    # Reduce dimensionality using UMAP
    umap = UMAP(n_components=3, random_state=42)
    clip_reduced = umap.fit_transform(clip_embeddings)
    dino_reduced = umap.fit_transform(dino_embeddings)

    # Visualize embeddings
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(clip_reduced[:, 0], clip_reduced[:, 1], clip_reduced[:, 2], c='red', label='CLIP')
    ax.scatter(dino_reduced[:, 0], dino_reduced[:, 1], dino_reduced[:, 2], c='blue', label='DINOv2')

    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_zlabel('UMAP 3')
    ax.legend()
    plt.title('Image Embeddings: CLIP vs DINOv2')
    plt.show()
    plt.savefig("plot.png")
    projector.visualize_embeddings()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir")
    args = parser.parse_args()
    main(args)