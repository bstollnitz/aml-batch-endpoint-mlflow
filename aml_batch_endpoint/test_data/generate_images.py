"""Generates images for inference."""

import logging
import os
from pathlib import Path

from torchvision import datasets

DATA_DIR = "aml_batch_endpoint/test_data/data"
IMAGES_DIR = "aml_batch_endpoint/test_data/images"
TEST_DATA_DIR = "aml_batch_endpoint/test_data"


def generate_images(num_images: int) -> None:
    """
    Generates images for inference.
    """
    test_data = datasets.FashionMNIST(DATA_DIR, train=False, download=True)

    images_dir = Path(IMAGES_DIR)
    if images_dir.exists():
        for f in Path(IMAGES_DIR).iterdir():
            f.unlink()
    else:
        os.makedirs(IMAGES_DIR)

    for i, (image, _) in enumerate(test_data):
        if i == num_images:
            break
        image.save(f"{IMAGES_DIR}/image_{i+1:0>3}.png")


def main():
    logging.basicConfig(level=logging.INFO)

    generate_images(20)


if __name__ == "__main__":
    main()
