import json
from pathlib import Path

import cv2
import faiss
from PIL import Image

from utils import embed_image, load_model, search_image

if __name__ == "__main__":
    DATASET_PATH = "final"
    RESULTS_NUM = 5
    device = "cuda"

    with open("index.json") as f:
        file_names = json.load(f)

    index = faiss.read_index("index.bin")
    model, preprocess = load_model(device)

    for img_path in Path(DATASET_PATH).glob("*.png"):
        img_idx = img_path.stem.split("_")[1]
        dst_folder = Path(f"nn/{img_idx}")
        dst_folder.mkdir(parents=True, exist_ok=True)

        D, I = search_image(img_path, index, model, preprocess)

        images = [cv2.imread(file_names[i]) for i in I[0]]
        scores = D[0]

        query_img = Image.open(img_path)
        query_img.save(dst_folder / "query.png")
        for i in range(len(images)):
            formatted_score = f"{scores[i]:.2f}".replace(".", ",")
            cv2.imwrite(str(dst_folder / f"{i}_d{formatted_score}.png"), images[i])
