import argparse
import json
from pathlib import Path

import faiss
from PIL import Image
from tqdm import tqdm

from utils import embed_image, index_data, load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=str)
    parser.add_argument(
        "--index-dir", type=str, help="If provided, extends an existing index"
    )
    parser.add_argument("--dst-dir", type=str, help="Where the index needs to be saved")
    parser.add_argument("--device", type=str)
    args = parser.parse_args()

    model, preprocess = load_model(args.device)

    src_index_dir = Path("indexes") / args.index_dir
    dst_idx_dir = Path("indexes") / args.dst_dir
    # img = Image.open("results/imgs/unconditional/4/render.jpg")
    # features = embed_image(img, preprocess)
    if args.index:
        index = faiss.read_index(src_index_dir / "index.bin")
        with open(src_index_dir / "index.json") as f:
            file_names = json.load(f)
    else:
        index = faiss.IndexFlatL2(768)
        file_names = []

    for img_path in tqdm(args.img_dir.glob("*.png")):
        img = Image.open(img_path)
        features = embed_image(img, preprocess)
        index = index_data(features, img_path, index)
        file_names.append(str(img_path.absolute()))

    index1_gpu = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)

    faiss.write_index(index, dst_idx_dir / "index.bin")
    with open(dst_idx_dir / "index.json", "w") as f:
        json.dump(file_names, f)
