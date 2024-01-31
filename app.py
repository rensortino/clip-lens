import argparse
import json
import shutil
from pathlib import Path

import cv2
import faiss
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image

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


@app.route("/", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return "No file uploaded", 400

    K = int(request.form["K"]) if "K" in request.form else 5

    file = request.files["image"]
    D, I = search_image(file, index, model, preprocess, nres=K)
    images = [cv2.imread(file_names[i]) for i in I[0]]
    scores = [d for d in D[0]]

    Path("static/.tmp").mkdir(exist_ok=True)
    query_img_path = ".tmp/query.png"
    Image.open(file).save(Path("static") / query_img_path)
    image_paths = [f".tmp/{i}.png" for i in range(len(images))]
    [
        Image.fromarray(img).save(Path("static") / image_paths[i])
        for i, img in enumerate(images)
    ]

    return render_template(
        "index.html",
        query_img_path=query_img_path,
        paths_and_scores=zip(image_paths, scores),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=str, required=True)
    args = parser.parse_args()

    file_names, index = read_index(args.index_dir)
    app.run()
