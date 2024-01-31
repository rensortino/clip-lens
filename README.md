# clip-search-engine

An image-to-image search engine using CLIP embeddings to find the most similar images to the provided one.
It is based on the following [guide](https://blog.roboflow.com/clip-image-search-faiss/)

## Installation

## Usage

First, we need to index all the images e want to search from. To do so, we use the `index.py` script. It takes as input a folder containing all the images we want to index (`--img-dir`) and saves the index under `index.bin`. This is a file containing all the CLIP embeddings of the images in the folder. It also save the `index.json` file, which contains the file names of all the indexed images, to retrieve them later for visualization.

```bash
python build_index.py --img-dir <path/to/images> 
```

Then, we can use the `search.py` script to search for the most similar images to the one provided. It takes as input the path to the image we want to search for (`--img-path`) and the number of results we want to retrieve (`--n-results`). It returns the `n_results` most similar images to the one provided.

```bash