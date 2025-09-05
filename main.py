import fiftyone as fo
import fiftyone.zoo as foz


dataset = foz.load_zoo_dataset("mnist")
test_split = dataset.match_tags("test")
print(test_split)


import cv2
import numpy as np

import fiftyone.brain as fob

# Construct a ``num_samples x num_pixels`` array of images
embeddings = np.array([
    cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel()
    for f in test_split.values("filepath")
])

# Compute 2D representation
results = fob.compute_visualization(
    test_split,
    embeddings=embeddings,
    num_dims=2,
    method="umap",
    brain_key="mnist_test",
    verbose=True,
    seed=51,
)

session = fo.launch_app(test_split)
session.wait()