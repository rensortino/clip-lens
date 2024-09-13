import json
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import tqdm

from model import EmbeddingModel

app = Flask(__name__)
dataset_dir = Path("OfficeHomeDataset_10072016")
styles = [dir.name for dir in dataset_dir.iterdir() if dir.is_dir()]
objects = [dir.name for dir in (dataset_dir / "Art").iterdir() if dir.is_dir()]

class NpEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Convert similarity matrix to list for JSON serialization
similarity_matrix = torch.load("heatmap.pt")  # Replace with your actual data
similarity_list = similarity_matrix.tolist()

# Dummy image paths (replace with your actual image paths or data)
image_dir = Path("static/.tmp")
global image_paths
image_paths = [ ("/" / path).as_posix() for path in sorted(image_dir.iterdir())]

def compute_similarity_matrix(embeddings):
    if embeddings.norm != 1.0: # Normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings @ embeddings.T

def process_images(model, images, start=0):
    embeddings = torch.tensor([], device=model.device)
    for i, image_path in tqdm(enumerate(images)):
        shutil.copy(image_path, image_dir / f"{i + start:06}.png")
        embedding = model(Image.open(image_path))
        embeddings = torch.cat([embeddings, embedding])
    return embeddings

def get_heatmap_plotly(similarity_matrix):
    # Create heatmap data
    heatmap_trace = go.Heatmap(z=similarity_matrix, colorscale='inferno', zmin=0, zmax=1, hoverinfo="text", text=similarity_matrix)
    heatmap_layout = go.Layout(title='Similarity Matrix Heatmap', width=400, height=400)
    heatmap_fig = go.Figure(data=[heatmap_trace], layout=heatmap_layout)
    return heatmap_fig

def get_scatter_plotly(embeddings, colors, labels=None):
    labels = ["image"] * embeddings.shape[0] if labels is None else labels
    scatter_trace = go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode='markers',
        marker=dict(size=10, opacity=0.6, color=colors),
        hoverinfo="text",
        # text=[f'Image {i}' for i in range(len(embeddings))]
        text=labels
    )
    scatter_layout = go.Layout(title='Image Embeddings Scatter Plot', width=400, height=400,)
    return go.Figure(data=[scatter_trace], layout=scatter_layout)

@app.route('/')
def index():
    return render_template('heatmap_and_scatter.html', styles=styles, objects=objects)

@app.route('/generate_visualizations', methods=['POST'])
def generate_visualizations():
    color1 = "red"
    color2 = "blue"
    model_id = request.form['modelId']
    display_limit = int(request.form['displayLimit'])

    model = EmbeddingModel(model_id)

    style1 = request.form['style1']
    style2 = request.form['style2']
    object1 = request.form['object1']
    object2 = request.form['object2']

    images1 = list((dataset_dir / style1 / object1).iterdir())[:display_limit // 2]
    images2 = list((dataset_dir / style2 / object2).iterdir())[:display_limit // 2]
    images = images1 + images2
    images = [path.as_posix() for path in images]
    
    # Save uploaded files and process images to get embeddings
    embeddings1 = process_images(model, images1)
    embeddings2 = process_images(model, images2, start=embeddings1.shape[0])
    embeddings  = torch.cat([embeddings1, embeddings2])
    
    # Generate similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings).cpu().numpy()
    
    # Reduce dimensionality for scatter plot
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(embeddings.cpu().numpy())
    umap = UMAP(n_neighbors=15, n_components=2, metric='cosine')
    umap_emb = umap.fit_transform(embeddings.cpu().numpy())

    heatmap_fig = get_heatmap_plotly(similarity_matrix)
    colors = [color1] * embeddings1.shape[0] + [color2] * embeddings2.shape[0]
    scatter_pca = get_scatter_plotly(pca_emb, colors, labels=images)
    scatter_umap = get_scatter_plotly(umap_emb, colors, labels=images)

    global image_paths
    image_paths = [ ("/" / path).as_posix() for path in sorted(image_dir.iterdir())]
    
    # Prepare the response
    response = {
        'heatmap': heatmap_fig.to_dict(),
        'scatter_pca': scatter_pca.to_dict(),
        'scatter_umap': scatter_umap.to_dict(),
        "image_paths": image_paths
    }
    
    return json.dumps(response, cls=NpEncoder)

@app.route('/get_image/<int:index>')
def get_image(index):
    image = image_paths[index]
    return jsonify({
        "image": image,
        })

@app.route('/get_image_pair/<int:row>/<int:col>')
def get_image_pair(row, col):
    image1 = image_paths[row]
    image2 = image_paths[col]
    return jsonify({
        "image1": image1,
        "image2": image2,
        })
    
if __name__ == '__main__':
    app.run(debug=True)