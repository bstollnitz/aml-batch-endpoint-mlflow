"""Utilities that help with scoring neural networks."""

import torch
from typing import Optional
import numpy as np


def predict(images: list[np.ndarray[np.uint8, np.dtype[np.uint8]]],
            model: Optional[torch.nn.Module],
            device: Optional[str] = "cpu") -> list[torch.Tensor]:
    """
    Makes a prediction for the whole dataset once.
    """
    if model is None:
        return []

    model.to(device)
    model.eval()

    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for x in images:
            tensor = torch.tensor(x).float().to(device)
            predictions.extend(_predict_one_batch(model, tensor))
    return predictions


def _predict_one_batch(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Makes a prediction for input x.
    """
    with torch.no_grad():
        y_prime = model(x)
        probabilities = torch.nn.functional.softmax(y_prime, dim=1)
        predicted_indices = probabilities.argmax(1)
    return predicted_indices.cpu().numpy()
