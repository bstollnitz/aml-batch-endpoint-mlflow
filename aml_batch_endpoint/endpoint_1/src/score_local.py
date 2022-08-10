"""Code that helps us test our neural network before deploying to the cloud."""

import logging
from pathlib import Path

import mlflow
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np

from utils_score_nn import predict

IMAGES_DIR = "aml_batch_endpoint/test_data/images"
MODEL_DIR = "aml_batch_endpoint/endpoint_1/model"


def main():
    model = mlflow.pytorch.load_model(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_paths = [f for f in Path(IMAGES_DIR).iterdir() if Path.is_file(f)]
    image_paths.sort()
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            x = np.array(image).reshape(1, -1)
            images.append(x)

    dataloader = DataLoader(images)
    predicted_indices = predict(device, dataloader, model)
    predictions = [
        FashionMNIST.classes[predicted_index]
        for predicted_index in predicted_indices
    ]

    logging.info("Predictions: %s", predictions)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
