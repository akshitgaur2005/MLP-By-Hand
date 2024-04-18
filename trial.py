import numpy as np

#data = np.random.random((1, 1)) # Data shape is samples x features
data = np.ones((2, 3))
labels = np.ones((2, 1))

print(f"Data:\n{data}")
print(f"Labels:\n{labels}")

def relu(x):
    return x * (x > 0)

def relu_back(x):
    return 1 * (x > 0)

class Dense():

    def __init__(self, in_feats, out_feats):
        self.weights = np.random.random((in_feats, out_feats)) # So that dot product can happen - (samples x features) x (features x neurons in next layer) = (samples x neurons in next layer)
        self.bias = np.random.random((1, out_feats))
        #self.layer = np.array([[1]])
        #self.bias = np.array([[0]])

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def backward(self, premult, input):
        #print(f"premult:\n{premult}")
        d_sigma = relu_back(self.forward(input))
        print(f"d_sigma: {d_sigma.shape}, input: {input.shape}, weights: {self.weights.shape}")
        #print(f"d_sigma:\n{d_sigma}")
        dw = premult * np.multiply(input, d_sigma.T)
        #print(f"d_sigma * input:\n{np.multiply(d_sigma, input)}")
        #print(f"dw:\n{dw}")
        db = premult * d_sigma
        #print(f"db:\n{db}")
        d_last = premult * premult * (d_sigma * self.weights)
        #print(f"d_sigma * self.layer:\n{np.multiply(d_sigma, self.weights.T).T}")
        #print(f"d_last:\n{d_last}")
        return dw, db, d_last
'''
d1 = Dense(1, 2)
pred = d1.forward(data)
print(f"Pred:\n{pred}")
print(f"Relu Pred:\n{relu(pred)}")

premult = 2 * (relu(pred) - labels)

dw, db, dlast = d1.backward(premult, data)


d2 = Dense(2, 1)
print("-" * 10 + "Layer 2" + "-" * 10)

premult = dlast
pred2 = d2.forward(pred)
print(f"Pred 2:\n{pred2}")
print(f"Relu Pred:\n{relu(pred2)}")
dw1, db1, dlast1 = d2.backward(premult, pred)

#print(f"dW:\n{dw}\ndb:\n{db}\ndlast:\n{dlast}")
'''

class Model():

    def __init__(self, neurons=[3, 2, 1]):
        self.layers = []
        for i in range(len(neurons) - 1):
            self.layers.append(Dense(neurons[i], neurons[i+1]))

    def forward(self, X):
        input = X
        print("-" * 10 + "Forward" + "-" * 10)
        for i in range(len(self.layers) - 1):
            print(f"i: {i}, input:\n{input}")
            input = relu(self.layers[i].forward(input))
        return self.layers[-1].forward(input)

    def optimise(self, X, y, lr):
        preds = self.forward(X)
        premult = 2 * np.mean(relu(preds) - y) # Premult is for chain rule, basically dJ/da_L-1

        for i in range(len(self.layers)):
            layer = self.layers[len(self.layers) - 1 - i]
            dw, db, premult = layer.backward(premult, X)
            layer.weights -= lr * dw
            layer.bias -= lr * db

        

model = Model([3, 4, 1])

print(f"Class Pred:\n{model.forward(data)}")
model.optimise(data, labels, 0.1)
