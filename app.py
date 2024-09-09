import argparse
import json
import os
import random
import shutil
import string
from pathlib import Path

import faiss
import numpy as np
import torch
from clip import clip
from flask import Flask, redirect, render_template, request, session, url_for, jsonify
from PIL import Image
from model import EmbeddingModel

from utils import load_model, search_image

app = Flask(__name__)
app.secret_key = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(32))

index_container_dir = "indexes"

tmp_dir = Path("static/.tmp")
tmp_dir.mkdir(exist_ok=True)


def cosine_sim(emb1, emb2):
    # Normalize embeddings and return cosine similarity
    norm_emb1 = emb1 / np.linalg.norm(emb1) 
    norm_emb2 = emb2 / np.linalg.norm(emb2) 
    
    return np.dot(norm_emb1, norm_emb2)

def read_index(index_dir):
    if not isinstance(index_dir, Path):
        index_dir = Path(index_container_dir, index_dir)

    with open(index_dir / "index.json") as f:
        file_names = json.load(f)
    index = faiss.read_index(str(index_dir / "index.bin"))
    return file_names, index

def save_tmp_img(file, name="query.png"):
    tmp_dir = Path("static/.tmp")
    tmp_dir.mkdir(exist_ok=True)
    query_img_path = tmp_dir / name
    Image.open(file).save(query_img_path)
    return query_img_path

global file_names
global index
global model
global preprocess

model, preprocess = load_model("cuda")

def get_session_info():
    index_folders = [f for f in os.listdir(index_container_dir) if os.path.isdir(os.path.join(index_container_dir, f))]
    selected_index = session.get('selected_index')
    return index_folders, selected_index


@app.route("/")
def home():
    index_folders, selected_index = get_session_info()
    return render_template("index.html", index_folders=index_folders, selected_index=selected_index)

@app.route("/similarity")
def similarity():
    return render_template("similarity.html")

@app.route("/set_index_folder", methods=["POST"])
def set_index_folder():
    selected_index = request.form['index_folder']
    session['selected_index'] = selected_index
    global file_names, index
    file_names, index = read_index(selected_index)
    return redirect(url_for("home"))


@app.route("/clear")
def clear():
    if Path("static/.tmp").exists():
        shutil.rmtree("static/.tmp")
    return redirect(url_for("home"))


@app.route("/knn", methods=["POST"])
def knn_search():
    if 'selected_index' not in session:
        return redirect(url_for('home'))
    
    if "image" not in request.files:
        return "No file uploaded", 400

    K = int(request.form["K"]) if "K" in request.form else 5

    file = request.files["image"]
    image = Image.open(file)
    global index
    D, I = search_image(image, index, model, preprocess, nres=K)

    image_paths = [file_names[i] for i in I[0]]
    scores = [d for d in D[0]]

    query_img_path = save_tmp_img(file)
    dst_image_paths = [f"{tmp_dir.as_posix()}/{i}.png" for i in range(len(image_paths))]
    [
        shutil.copy(img, dst_image_paths[i])
        for i, img in enumerate(image_paths)
    ]

    # index_folders, selected_index = get_session_info()
    return render_template(
        "knn.html",
        query_img_path=query_img_path,
        paths_and_scores=zip(dst_image_paths, scores),
        # index_folders=index_folders,
        # selected_index=selected_index,
    )


@app.route("/clip", methods=["POST"])
def clip_similarity():
    if 'selected_index' not in session:
        return redirect(url_for('home'))
    
    if "image1" not in request.files or "image2" not in request.files:
        return "Two images required", 400

    query_image = request.files["image1"]
    query_image_path = save_tmp_img(query_image, "query.png")
    query_image = Image.open(query_image)
    query_image = preprocess(query_image).unsqueeze(0).to("cuda")

    comparison_image = request.files["image2"]
    comparison_image_path = save_tmp_img(comparison_image, "comparison.png")
    comparison_image = Image.open(comparison_image)
    comparison_image = preprocess(comparison_image).unsqueeze(0).to("cuda")

    with torch.no_grad():
        query_embedding = model.encode_image(query_image).squeeze().cpu().numpy()
        comparison_embedding = model.encode_image(comparison_image).squeeze().cpu().numpy() 

    query_embedding = query_embedding / np.linalg.norm(query_embedding) 
    comparison_embedding = comparison_embedding / np.linalg.norm(comparison_embedding) 
    
    similarity = np.dot(query_embedding, comparison_embedding)

    index_folders, selected_index = get_session_info()
    return render_template(
        "index.html",
        tab="clip",
        similarity=similarity,
        query_image_path=query_image_path,
        comparison_image_path=comparison_image_path,
        index_folders=index_folders,
        selected_index=selected_index,
    )

@app.route("/latent_similarity", methods=["POST"]) # TODO Refactor
def latent_similarity():
    if "image1" not in request.files:
        if "query_image" not in session:
            return "Query Image not defined"
        query_image = session.get("query_image")
    else:
        query_image = request.files["image1"]

    query_image_path = save_tmp_img(query_image, "query.png")
    session['query_image'] = query_image_path.as_posix()
    query_image = Image.open(query_image)
    query_image = preprocess(query_image).unsqueeze(0).to("cuda")

    if "image2" not in request.files:
        if "comparison_image" not in session:
            return "Query Image not defined"
        comparison_image = session.get("comparison_image")
    else:
        comparison_image = request.files["image2"]

    # comparison_image = request.files["image2"]
    comparison_image_path = save_tmp_img(comparison_image, "comparison.png")
    session['comparison_image'] = comparison_image_path.as_posix()
    comparison_image = Image.open(comparison_image)
    comparison_image = preprocess(comparison_image).unsqueeze(0).to("cuda")

    model_id = request.form["model-name"]
    distance_metric = request.form["distance-metric"]
    
    model = EmbeddingModel(model_id)

    with torch.no_grad():
        query_embedding = model.encode_image(query_image).squeeze().cpu().numpy()
        comparison_embedding = model.encode_image(comparison_image).squeeze().cpu().numpy() 

    if distance_metric == "cosine":
        similarity = cosine_sim(query_embedding, comparison_embedding)
    elif distance_metric == "euclidean":
        similarity = np.linalg.norm(query_embedding - comparison_embedding)

    return jsonify({
        'similarity': float(similarity),
        # 'query_image_path': query_image_path.as_posix(),
        # 'comparison_image_path': comparison_image_path.as_posix()
    })

    # return render_template(
    #     "similarity.html",
    #     similarity=similarity,
    #     query_image_path=query_image_path,
    #     comparison_image_path=comparison_image_path,
    # )

@app.route("/clip_multiple", methods=["POST"])
def clip_similarity_multiple():
    if 'selected_index' not in session:
        return redirect(url_for('home'))
    
    if "image1" not in request.files or "image2" not in request.files:
        return "Two images required", 400

    query_image = request.files["image1"]
    query_image_path = save_tmp_img(query_image)
    query_image = Image.open(query_image)
    query_image = preprocess(query_image).unsqueeze(0).to("cuda")

    comparison_dir = request.files["image2"]

    similarities = []
    dst_image_paths = []

    with torch.no_grad():
        query_embedding = model.encode_image(query_image).squeeze().cpu().numpy()
    query_embedding = query_embedding / np.linalg.norm(query_embedding) 
    
    for i, path in enumerate(os.listdir(comparison_dir)):
        image2 = Image.open(path)
        dst_image_paths.append(f"{tmp_dir.as_posix()}/{i}.png")
        
        image2 = preprocess(image2).unsqueeze(0).to("cuda")
        with torch.no_grad():
            embedding2 = model.encode_image(image2).squeeze().cpu().numpy() / np.linalg.norm(embedding2)
        
        similarity = np.dot(query_embedding, embedding2)
        similarities.append(similarity)


    # index_folders, selected_index = get_session_info()
    return render_template(
        "comparison.html",
        query_image_path=query_image_path,
        paths_and_scores=zip(dst_image_paths, similarities),
        # index_folders=index_folders,
        # selected_index=selected_index,
    )


@app.route("/text_index", methods=["POST"])
def text_index():
    if 'selected_index' not in session:
        return redirect(url_for('home'))
    
    if "text" not in request.form:
        return "Text required", 400

    text = request.form["text"]
    text_tokens = clip.tokenize([text]).to("cuda")
    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens).cpu().numpy().astype('float32')

    faiss.normalize_L2(text_embedding)
    D, I = index.search(text_embedding.reshape(1, -1), 5)

    image_paths = [file_names[i] for i in I[0]]
    scores = [d for d in D[0]]

    dst_image_paths = [f"{tmp_dir.as_posix()}/{i}.png" for i in range(len(image_paths))]
    [
        shutil.copy(img, dst_image_paths[i])
        for i, img in enumerate(image_paths)
    ]

    # dst_image_paths = ["/".join(path.split('/')[1:]) for path in dst_image_paths]

    # index_folders, selected_index = get_session_info()

    return render_template(
        "text_index.html",
        text_prompt=text,
        paths_and_scores=zip(dst_image_paths, scores),
        # index_folders=index_folders,
        # selected_index=selected_index,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=str, required=True)
    args = parser.parse_args()

    app.run()