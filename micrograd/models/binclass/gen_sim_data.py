import numpy as np
import os

# Generate synthetic data for two classes
class_0 = np.random.randn(50, 2) + np.array([1, 1])
class_1 = np.random.randn(50, 2) + np.array([3, 3])

# Labels
data = np.vstack([class_0, class_1])
labels = np.array([0] * 50 + [1] * 50).reshape(-1, 1)

# Save data
cur_dir = os.path.dirname(__file__)
data_path = os.path.join(cur_dir, "data.npy")
labels_path = os.path.join(cur_dir, "labels.npy")
np.save(data_path, data)
np.save(labels_path, labels)
