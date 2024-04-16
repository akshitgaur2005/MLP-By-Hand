import numpy as np

data = np.random.random((1, 3)) # Data shape is samples x features
labels = np.ones((1, 1))

print(f"Data:\n{data}")
print(f"Labels:\n{labels}")

def relu(x):
    return x * (x > 0)

def relu_back(x):
    return 1 * (x > 0)

class Dense():

    def __init__(self, in_feats, out_feats):
        self.layer = np.random.random((in_feats, out_feats)) # So that dot product can happen - (samples x features) x (features x neurons in next layer) = (samples x neurons in next layer)
        self.bias = np.random.random((1, out_feats))

    def forward(self, X):
        return np.dot(X, self.layer) + self.bias

    def backward(self, premult, input):
        print(f"premult:\n{premult}")
        d_sigma = relu_back(self.forward(input))
        print(f"d_sigma:\n{d_sigma}")
        dw = premult * np.multiply(d_sigma, input)
        print(f"d_sigma * input:\n{np.multiply(d_sigma, input)}")
        print(f"dw:\n{dw}")
        db = premult * d_sigma
        print(f"db:\n{db}")
        d_last = premult * (d_sigma * self.layer.T).T
        print(f"d_sigma * self.layer:\n{np.multiply(d_sigma, self.layer.T).T}")
        print(f"d_last:\n{d_last}")
        return dw, db, d_last

d1 = Dense(3, 1)
pred = d1.forward(data)
print(f"Pred:\n{pred}")
print(f"Relu Pred:\n{relu(pred)}")

premult = 2 * (relu(pred) - labels)

dw, db, dlast = d1.backward(premult, data)
#print(f"dW:\n{dw}\ndb:\n{db}\ndlast:\n{dlast}")
'''
class Model():

    def __init__(self, neurons=[3, 2, 1]):
        self.layers = []
        for i in range(len(neurons) - 1):
            self.layers.append(Dense(neurons[i], neurons[i+1]))

    def forward(self, X):
        input = X
        for layer in self.layers:
            input = layer.forward(input)
            input = relu(input)
        return input

model = Model([3, 4, 1])

print(f"Class Pred:\n{model.forward(data)}")
'''
