# clip-search-engine

An image-to-image search engine using CLIP embeddings to find the most similar images to the provided one.
It is based on [this guide](https://blog.roboflow.com/clip-image-search-faiss/)

<div align="center">
  <img src="static/logo.jpeg" alt="logo" width="600"/>
</div>

## Installation

Install the necessary requirements from the `requirements.txt` file. This repo uses [faiss](https://github.com/facebookresearch/faiss) to build the index and search for the nearest neighbors, and [Flask](https://flask.palletsprojects.com/en/3.0.x/) to visualize the results in a webpage.
```bash
pip install -r requirements.txt
```

## Usage

First, we need to index all the images we want to search from. To do so, we use the `index.py` script. It takes as input a folder containing all the images we want to index (`--img-dir`) and where to save the output files (`--dst-dir`). It then saves the index, i.e., the file containing all the CLIP embeddings of the images in the folder, under `index.bin`. It also saves the `index.json` file, which contains the file names of all the indexed images, to retrieve them later for visualization.

```bash
python build_index.py --img-dir <path/to/images> --dst-dir <path/to/dir> 
```

Then, we can run the Flask app to load a basic web page to load the query image and visualize the nearest neighbor. The only parameter to specify is the folder containing the index (`--index-dir`)

```bash
python app.py --index-dir <path/to/dir> 
```

As an alternative, we can use the `query.py` script to procedurally get the nearest neighbor images for all the images in a folder. The images should be save in `.png` under the folder specified by `--img-dir`.

```bash
python query.py --img-dir <path/to/query/images> --index-dir <path/to/index> --nres <K>
```
