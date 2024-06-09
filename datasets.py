import json
from sklearn.datasets import make_blobs, make_moons, make_circles
import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(
    dataset_type="circles",
    savefile="data.json",
    points_count=100,
    noise=0.1,
    no_centres=3,
    to_plot=True,
):
    if dataset_type == "circles":
        X, y = make_circles(
            n_samples=points_count, noise=noise, factor=0.5, random_state=41
        )
    elif dataset_type == "moons":
        X, y = make_moons(n_samples=points_count, noise=noise, random_state=41)
    elif dataset_type == "blobs":
        X, y = make_blobs(n_samples=points_count, centers=no_centres, random_state=41)
    else:
        raise ValueError(
            "Unsupported dataset type. Choose from 'circles', 'moons', or 'blobs'."
        )

    points = [(float(x), float(y)) for x, y in X]
    return points
