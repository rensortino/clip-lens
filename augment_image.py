from pathlib import Path
from PIL import Image
from tqdm import tqdm


image_path = Path("OfficeHomeDataset_10072016/Clipart/Bucket/00004.jpg")
img = Image.open(image_path)
out_dir = Path(f"static/rotations/{image_path.stem}")
out_dir.mkdir(exist_ok=True, parents=True)
for angle in tqdm(range(0,360)):
    img.rotate(angle).save(out_dir / f"{angle:06}.png")