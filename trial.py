import numpy as np

data = np.random.random((4, 3))
print(data)

class Dense():

    def __init__(self, in_feats, out_feats):
        self.layer = np.random.random((in_feats, out_feats))
        self.bias = np.random.random((out_feats))

    def forward(self, X):
        return np.dot(X, self.layer) + self.bias

dense1 = Dense(3, 2)
dense2 = Dense(2, 1)

preds1 = dense1.forward(data)
print(f"preds1: {preds1}")
preds2 = dense2.forward(preds1)
print(f"preds2: {preds2}")
