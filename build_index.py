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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--distance", type=str, default="cosine")
    args = parser.parse_args()

    model, preprocess = load_model(args.device)

    src_index_dir = Path("indexes") / args.index_dir if args.index_dir is not None else None
    dst_idx_dir = Path("indexes") / args.dst_dir
    dst_idx_dir.mkdir(exist_ok=True, parents=True)
    

    img_dir = Path(args.img_dir)
    assert img_dir.exists(), f"Image dir not found at {img_dir}"
    if args.index_dir:
        index = faiss.read_index(src_index_dir / "index.bin")
        with open(src_index_dir / "index.json") as f:
            file_names = json.load(f)
    else:
        res = faiss.StandardGpuResources()
        if args.distance == "cosine":
            index = faiss.GpuIndexFlatIP(res, 768)
        elif args.distance == "euclidean":
            index = faiss.GpuIndexFlatL2(res, 768)

        file_names = []

    for img_path in tqdm(img_dir.glob("**/*.jpg")):
        img = Image.open(img_path)
        features = embed_image(img, model, preprocess)
        index = index_data(features, img_path, index)
        file_names.append(str(img_path.absolute()))

    index_cpu = faiss.index_gpu_to_cpu(index)

    faiss.write_index(index_cpu, (dst_idx_dir / "index.bin").as_posix())
    with open(dst_idx_dir / "index.json", "w") as f:
        json.dump(file_names, f)
