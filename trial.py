import numpy as np

data = np.random.random((4, 3))
print(data)

def relu(x):
    if (x >= 0):
        return x
    else:
        return 0

class Dense():

    def __init__(self, in_feats, out_feats):
        self.layer = np.random.random((in_feats, out_feats))
        self.bias = np.random.random((out_feats))

    def forward(self, X):
        return np.dot(X, self.layer) + self.bias

dense1 = Dense(3, 2)
dense2 = Dense(2, 1)

print(f"Pred: {dense2.forward(dense1.forward(data))}")


class Model():

    def __init__(self, neurons=[3, 2, 1]):
        self.layers = []
        for i in range(len(neurons) - 1):
            self.layers.append(Dense(neurons[i], neurons[i+1]))

    def forward(self, X):
        input = X
        for layer in self.layers:
            input = layer.forward(input)
        return input

model = Model()

print(f"Class Pred: {model.forward(data)}")
