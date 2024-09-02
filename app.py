import argparse
import json
import shutil
from pathlib import Path

import faiss
import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
import torch
from clip import clip

from utils import load_model, search_image

app = Flask(__name__)


def read_index(index_dir):
    if not isinstance(index_dir, Path):
        index_dir = Path(index_dir)

    with open(index_dir / "index.json") as f:
        file_names = json.load(f)
    index = faiss.read_index(str(index_dir / "index.bin"))
    return file_names, index


global file_names
global index
global model
global preprocess

model, preprocess = load_model("cuda")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/clear")
def clear():
    if Path("static/.tmp").exists():
        shutil.rmtree("static/.tmp")
    return redirect(url_for("home"))


@app.route("/knn", methods=["POST"])
def knn_search():
    if "image" not in request.files:
        return "No file uploaded", 400

    K = int(request.form["K"]) if "K" in request.form else 5

    file = request.files["image"]
    D, I = search_image(file, index, model, preprocess, nres=K)

    image_paths = [file_names[i] for i in I[0]]
    scores = [d for d in D[0]]

    tmp_dir = Path("static/.tmp")
    tmp_dir.mkdir(exist_ok=True)
    query_img_path = tmp_dir / "query.png"
    Image.open(file).save(query_img_path)
    dst_image_paths = [f"{tmp_dir.as_posix()}/{i}.png" for i in range(len(image_paths))]
    [
        shutil.copy(img, dst_image_paths[i])
        for i, img in enumerate(image_paths)
    ]

    return render_template(
        "index.html",
        tab="knn",
        query_img_path=query_img_path,
        paths_and_scores=zip(dst_image_paths, scores),
    )


@app.route("/clip", methods=["POST"])
def clip_similarity():
    if "image1" not in request.files or "image2" not in request.files:
        return "Two images required", 400

    image1 = Image.open(request.files["image1"])
    image2 = Image.open(request.files["image2"])

    image1 = preprocess(image1).unsqueeze(0).to("cuda")
    image2 = preprocess(image2).unsqueeze(0).to("cuda")

    with torch.no_grad():
        embedding1 = model.encode_image(image1).squeeze().cpu().numpy()
        embedding2 = model.encode_image(image2).squeeze().cpu().numpy()

    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    return render_template(
        "index.html",
        tab="clip",
        similarity=similarity,
    )


@app.route("/text_index", methods=["POST"])
def text_index():
    if "text" not in request.form:
        return "Text required", 400

    text = request.form["text"]
    text_tokens = clip.tokenize([text]).to("cuda")
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).squeeze().cpu().numpy()

    faiss.normalize_L2(text_embedding.astype('float32'))
    D, I = index.search(text_embedding.reshape(1, -1), 5)

    image_paths = [file_names[i] for i in I[0]]
    scores = [d for d in D[0]]

    tmp_dir = Path("static/.tmp")
    tmp_dir.mkdir(exist_ok=True)
    dst_image_paths = [f"{tmp_dir.as_posix()}/{i}.png" for i in range(len(image_paths))]
    [
        shutil.copy(img, dst_image_paths[i])
        for i, img in enumerate(image_paths)
    ]

    dst_image_paths = ["/".join(path.split('/')[1:]) for path in dst_image_paths]

    return render_template(
        "index.html",
        tab="text_index",
        paths_and_scores=zip(dst_image_paths, scores),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=str, required=True)
    args = parser.parse_args()

    file_names, index = read_index(args.index_dir)
    app.run()