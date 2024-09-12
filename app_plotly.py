import json
import shutil
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from sklearn.decomposition import PCA
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
# image_paths = [ ("/" / path).as_posix() for path in sorted(Path("static/.tmp").iterdir())]
image_dir = Path("static/.tmp")
global image_paths
image_paths = [ ("/" / path).as_posix() for path in sorted(image_dir.iterdir())]

def compute_similarity_matrix(embeddings):
    if embeddings.norm != 1.0: # Normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings @ embeddings.T

def process_images(model, images):
    embeddings = torch.tensor([], device=model.device)
    for i, image_path in tqdm(enumerate(images)):
        shutil.copy(image_path, image_dir / f"{i:06}.png")
        embedding = model(Image.open(image_path))
        embeddings = torch.cat([embeddings, embedding])
    return embeddings

@app.route('/')
def index():
    return render_template('heatmap_and_scatter.html', styles=styles, objects=objects)

@app.route('/generate_visualizations', methods=['POST'])
def generate_visualizations():
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
    embeddings = process_images(model, images)
    
    # Generate similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings).cpu().numpy()
    
    # Reduce dimensionality for scatter plot
    pca = PCA(n_components=2)
    # or: umap = UMAP(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings.cpu().numpy())
    
    # Create heatmap data
    heatmap_trace = go.Heatmap(z=similarity_matrix, colorscale='inferno', zmin=0, zmax=1)
    heatmap_layout = go.Layout(title='Similarity Matrix Heatmap')
    
    colors = ["#000000"]*reduced_embeddings.shape[0],

    # Create scatter plot data
    scatter_trace = go.Scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        
        mode='markers',
        marker=dict(size=10, opacity=0.6, color=colors),
        text=[f'Image {i}' for i in range(len(reduced_embeddings))]
    )
    scatter_layout = go.Layout(title='Image Embeddings Scatter Plot')

    heatmap_fig = go.Figure(data=[heatmap_trace], layout=heatmap_layout)
    scatter_fig = go.Figure(data=[scatter_trace], layout=scatter_layout)

    global image_paths
    # image_paths = [path.as_posix() for path in ("/" / image_dir).iterdir()]
    image_paths = [ ("/" / path).as_posix() for path in sorted(image_dir.iterdir())]
    
    # Prepare the response
    response = {
        'heatmap': heatmap_fig.to_dict(),
        'scatter': scatter_fig.to_dict(),
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