import csv
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorboard.plugins import projector


class EmbeddingProjector():
    def __init__(self, logdir="logdir"):
        self.logdir = logdir
        self._setup_tb_logdir()
        self._setup_tb_sprite_dir()

        # Set up config for projector
        self.config = projector.ProjectorConfig()
        self.embedding = self.config.embeddings.add()
        self.embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        self.embedding.metadata_path = 'metadata.tsv'  # Optional: path to metadata file

    # Set up a logs directory for TensorBoard
    def _setup_tb_logdir(self):
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    def _setup_tb_sprite_dir(self, dir_name="sprites"):
        self.sprite_dir = os.path.join(self.logdir, dir_name)
        if not os.path.exists(self.sprite_dir):
            os.makedirs(self.sprite_dir)

    def generate_sprite_images(self, image_dir, sprite_size=32):
        # Generate sprite image
        image_paths = list(os.listdir(image_dir))
        num_images = len(image_paths)
        num_rows = int(np.ceil(np.sqrt(num_images)))
        sprite_image = Image.new(mode='RGB', size=(sprite_size*num_rows, sprite_size*num_rows), color=(255,255,255))

        for i, img_path in enumerate(image_paths):
            try:
                img_path = Path(image_dir) / img_path
                img = Image.open(img_path).convert("RGB")
                img = img.resize((sprite_size, sprite_size), Image.LANCZOS)
                sprite_image.paste(img, box=(sprite_size * (i % num_rows), sprite_size * (i // num_rows)))
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # Use a blank image if there's an error
                sprite_image.paste(Image.new('RGB', (sprite_size, sprite_size), color='white'), 
                                box=(sprite_size * (i % num_rows), sprite_size * (i // num_rows)))

        sprite_path = os.path.join(self.sprite_dir, 'sprite.jpg')
        sprite_image.save(sprite_path)
        # Set up the sprite and metadata for the projector
        self.embedding.sprite.image_path = os.path.relpath(sprite_path, self.logdir)
        self.embedding.sprite.single_image_dim.extend([sprite_size, sprite_size])
        self._write_metadata(image_paths)

    def _write_metadata(self, image_paths):
        # Generate metadata file (optional)
        with open(os.path.join(self.logdir, 'metadata.tsv'), 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            # writer.writerow(['Image Path'])  # Header
            for path in image_paths:
                writer.writerow([path])

    def save_embeddings(self, embeddings):
        weights = tf.Variable(embeddings)
        # Create a checkpoint from embedding, the filename and key are the
        # name of the tensor.
        checkpoint = tf.train.Checkpoint(embedding=weights)
        checkpoint.save(os.path.join(self.logdir, "embedding.ckpt"))

    def visualize_embeddings(self):
        # Save the projector config
        projector.visualize_embeddings(self.logdir, self.config)
        print(f"Embeddings saved. To view, run: tensorboard --logdir={log_dir}")
        print("Then, open the provided URL in your web browser and navigate to the 'Projector' tab.")