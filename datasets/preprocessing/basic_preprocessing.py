from sklearn import preprocessing
import numpy as np
import torch

class LabelOneHotEncoder(object):

    def __call__(self, data):
        assert len(data.shape) == 1, "Labels must have only one dimension"
        data = data.astype(np.int32)
        encoded = np.zeros((data.size, data.max() + 1))
        encoded[np.arange(data.size), data] = 1
        return encoded


# Transformers
class LabelEncoder(object):
    def __init__(self):
        # Initialize an instance of the LabelEncoder from scikit-learn
        self.le = preprocessing.LabelEncoder()

    def __call__(self, data):
        # Fit the LabelEncoder to the provided data
        self.le.fit(data)
        # Transform the data using the fitted LabelEncoder and convert to int32
        data = self.le.transform(data).astype("int32")
        # Store the unique target names after encoding
        self.target_names = list(self.le.classes_)
        # Store the number of unique classes after encoding
        self.n_classes = len(self.le.classes_)
        return data

class ToTensor(object):
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, data):
        data = np.array(data, dtype=np.float32)
        return torch.from_numpy(data).to(dtype=self.dtype)

class LabelFlatten(object):
    def __call__(self, data):
        return data.view(-1, 1)