import json
import shutil
from pathlib import Path

import numpy as np
from jinja2 import Template
import plotly.graph_objects as go
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from sklearn.decomposition import PCA
from umap import UMAP
from tqdm import tqdm

from model import EmbeddingModel, CaptioningModel

app = Flask(__name__)
dataset_dir = Path("DomainNet")
styles = [dir.name for dir in sorted(dataset_dir.iterdir()) if dir.is_dir()]
objects = [dir.name for dir in sorted((dataset_dir / "sketch").iterdir()) if dir.is_dir()]


class NpEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


image_dir = Path("static/.tmp")
global image_paths
image_paths = [("/" / path).as_posix() for path in sorted(image_dir.iterdir())]

global captions
captions = []

global figures
figures = {}


def compute_similarity_matrix(embeddings):
    if embeddings.norm != 1.0:  # Normalize
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings @ embeddings.T


def process_images(model, images, start=0):
    embeddings = torch.tensor([], device=model.device)
    for i, image_path in tqdm(enumerate(images)):
        shutil.copy(image_path, image_dir / f"{i + start:06}.png")
        embedding = model(Image.open(image_path))
        embeddings = torch.cat([embeddings, embedding])
    return embeddings


def generate_captions(model, images, start=0):
    captions = []
    for i, image_path in tqdm(enumerate(images)):
        caption = model(Image.open(image_path))
        captions.append(caption)
    return captions


def get_heatmap_plotly(similarity_matrix):
    heatmap_trace = go.Heatmap(
        z=similarity_matrix,
        colorscale="inferno",
        showscale=False,
        zmin=0,
        zmax=1,
        hoverinfo="text",
        text=[[f"{value:.4f}" for value in row] for row in similarity_matrix],
    )

    heatmap_layout = go.Layout(
        title="Similarity Matrix Heatmap",
        width=400,
        height=400,
        xaxis=dict(showspikes=True),
        yaxis=dict(showspikes=True),
    )
    return go.Figure(data=[heatmap_trace], layout=heatmap_layout)


def get_scatter_plotly(embeddings, colors, labels=None):
    labels = ["image"] * embeddings.shape[0] if labels is None else labels
    scatter_trace = go.Scatter(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        mode="markers",
        marker=dict(size=12, opacity=0.6, color=colors),
        hoverinfo="text",
        showlegend=True,
        text=labels,
    )
    scatter_layout = go.Layout(
        title="Image Embeddings Scatter Plot",
        width=400,
        height=400,
    )
    return go.Figure(data=[scatter_trace], layout=scatter_layout)


@app.route("/")
def index():
    return render_template("heatmap_and_scatter.html", styles=styles, objects=objects)


@app.route("/generate_visualizations", methods=["POST"])
def generate_visualizations():
    color1 = "red"
    color2 = "blue"
    model_id = request.form["modelId"]
    display_limit = int(request.form["displayLimit"])

    model = EmbeddingModel(model_id)

    style1 = request.form["style1"]
    style2 = request.form["style2"]
    object1 = request.form["object1"]
    object2 = request.form["object2"]

    images1 = list((dataset_dir / style1 / object1).iterdir())[: display_limit // 2]
    images2 = list((dataset_dir / style2 / object2).iterdir())[: display_limit // 2]
    images = images1 + images2
    images = [path.as_posix() for path in images]

    # Save uploaded files and process images to get embeddings
    embeddings1 = process_images(model, images1)
    embeddings2 = process_images(model, images2, start=embeddings1.shape[0])
    embeddings = torch.cat([embeddings1, embeddings2])

    # Generate similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings).cpu().numpy()

    # Reduce dimensionality for scatter plot
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(embeddings.cpu().numpy())
    umap = UMAP(n_neighbors=15, n_components=2, metric="cosine")
    umap_emb = umap.fit_transform(embeddings.cpu().numpy())

    heatmap_fig = get_heatmap_plotly(similarity_matrix)

    colors = [color1] * embeddings1.shape[0] + [color2] * embeddings2.shape[0]
    labels = [",".join([path.split("/")[-2], path.split("/")[1]]) for path in images]
    scatter_pca = get_scatter_plotly(pca_emb, colors, labels=labels)
    scatter_umap = get_scatter_plotly(umap_emb, colors, labels=labels)

    global figures
    figures["heatmap"] = heatmap_fig
    figures["scatter_pca"] = scatter_pca
    figures["scatter_umap"] = scatter_umap

    global image_paths
    image_paths = [("/" / path).as_posix() for path in sorted(image_dir.iterdir())]

    # Prepare the response
    response = {
        "heatmap": heatmap_fig.to_dict(),
        "scatter_pca": scatter_pca.to_dict(),
        "scatter_umap": scatter_umap.to_dict(),
        "image_paths": image_paths,
    }

    return json.dumps(response, cls=NpEncoder)

@app.route("/generate_visualizations_captions", methods=["POST"])
def generate_visualizations_captions(): #TODO Refactor functions, don't repeat for 1 and 2
    color1 = "red"
    color2 = "blue"
    model_id = request.form["modelId"]
    display_limit = int(request.form["displayLimit"])

    model = EmbeddingModel(model_id)
    captioning_model = CaptioningModel("Salesforce/blip-image-captioning-base")

    style1 = request.form["style1"]
    style2 = request.form["style2"]
    object1 = request.form["object1"]
    object2 = request.form["object2"]

    images1 = list((dataset_dir / style1 / object1).iterdir())[: display_limit // 2]
    images2 = list((dataset_dir / style2 / object2).iterdir())[: display_limit // 2]
    images = images1 + images2
    images = [path.as_posix() for path in images]

    captions1 = generate_captions(captioning_model, images1)
    captions2 = generate_captions(captioning_model, images2, start=len(captions1))
    global captions
    captions = captions1 + captions2

    text_emb1 = model.encode_text(captions1)
    text_emb2 = model.encode_text(captions2)

    # Save uploaded files and process images to get embeddings
    img_emb1 = process_images(model, images1)
    img_emb2 = process_images(model, images2, start=img_emb1.shape[0])

    emb1 = torch.cat([img_emb1, text_emb1], dim=1)
    emb2 = torch.cat([img_emb2, text_emb2], dim=1)
    embeddings = torch.cat([emb1, emb2])

    # Generate similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings).cpu().numpy()

    # Reduce dimensionality for scatter plot
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(embeddings.cpu().numpy())
    umap = UMAP(n_neighbors=15, n_components=2, metric="cosine")
    umap_emb = umap.fit_transform(embeddings.cpu().numpy())

    heatmap_fig = get_heatmap_plotly(similarity_matrix)

    colors = [color1] * emb1.shape[0] + [color2] * emb2.shape[0]
    labels = [",".join([path.split("/")[-2], path.split("/")[1]]) for path in images]
    scatter_pca = get_scatter_plotly(pca_emb, colors, labels=labels)
    scatter_umap = get_scatter_plotly(umap_emb, colors, labels=labels)

    global figures
    figures["heatmap"] = heatmap_fig
    figures["scatter_pca"] = scatter_pca
    figures["scatter_umap"] = scatter_umap

    global image_paths
    image_paths = [("/" / path).as_posix() for path in sorted(image_dir.iterdir())]

    # Prepare the response
    response = {
        "heatmap": heatmap_fig.to_dict(),
        "scatter_pca": scatter_pca.to_dict(),
        "scatter_umap": scatter_umap.to_dict(),
        "image_paths": image_paths,
        "captions": captions,
    }

    return json.dumps(response, cls=NpEncoder)


# @app.route("/get_image/<int:index>")
# def get_image(index):
#     image = image_paths[index]
#     return jsonify(
#         {
#             "image": image,
#         }
#     )


@app.route("/get_image_pair/<int:row>/<int:col>")
def get_image_pair(row, col):
    image1 = image_paths[row]
    image2 = image_paths[col]
    caption1 = captions[row] if len(captions) > row else ''
    caption2 = captions[col] if len(captions) > col else ''
    return jsonify(
        {
            "image1": image1,
            "image2": image2,
            "caption1": caption1,
            "caption2": caption2,
        }
    )


@app.route("/export")
def export_plotly():
    output_html_path = r"plots/out.html"
    input_template_path = r"templates/export.html"

    global figures
    plotly_jinja_data = {
        fig_name: fig_data.to_html(full_html=False)
        for fig_name, fig_data in figures.items()
    }

    try:
        with open(output_html_path, "w", encoding="utf-8") as output_file:
            with open(input_template_path) as template_file:
                j2_template = Template(template_file.read())
                output_file.write(j2_template.render(plotly_jinja_data))
        return "Plots exported correctly"
    except:
        return "Error when saving plots"


if __name__ == "__main__":
    app.run(debug=True)
