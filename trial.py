import numpy as np

#data = np.random.random((1, 1)) # Data shape is samples x features
#data = np.ones((2, 3))
#labels = np.ones((2, 1))
data = np.array([[1, 0, 1],
                 [0, 1, 1]])
labels = np.array([[1],
                   [0]])

print(f"Data:\n{data}")
print(f"Labels:\n{labels}")

def relu(x):
    return x * (x > 0)

def relu_back(x):
    return 1 * (x > 0)

def cost(preds, y):
    return np.mean(np.square(preds - y))

class Dense():

    def __init__(self, in_feats, out_feats):
        self.weights = np.random.randn(in_feats, out_feats) # So that dot product can happen - (samples x features) x (features x neurons in next layer) = (samples x neurons in next layer)
        self.bias = np.random.randn(1, out_feats)
        #self.layer = np.array([[1]])
        #self.bias = np.array([[0]])

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def backward(self, da_i, a_i_1):
        dw_i = np.dot(a_i_1.T, np.multiply(da_i, relu_back(self.forward(a_i_1))))
        db_i = np.multiply(da_i, relu_back(self.forward(a_i_1)))
        da_i_1 = np.dot(np.multiply(da_i, relu(self.forward(a_i_1))), self.weights.T)
        return dw_i, db_i, da_i_1

class Model():

    def __init__(self, neurons=[3, 2, 1]):
        self.layers = []
        for i in range(len(neurons) - 1):
            self.layers.append(Dense(neurons[i], neurons[i+1]))

    def forward(self, X, opt=False):
        input = X
        #print("-" * 10 + "Forward" + "-" * 10)
        if opt:
            outputs = []
        for i in range(len(self.layers) - 1):
            #print(f"i: {i}, input:\n{input}")
            input = relu(self.layers[i].forward(input))
            if opt:
                outputs.append(input)
        preds = self.layers[-1].forward(input)
        if opt:
            outputs.append(preds)
            return outputs
        return preds

    def optimise(self, X, y, lr):
        preds = self.forward(X, opt=True)
        a_L = preds[-1]
        da_i_1 = (a_L - y) / y.shape[1]
        inputs = preds

        for i in range(len(self.layers)):
            j = len(self.layers) - 1 - i
            if j > 0:
                a_i_1 = inputs[j - 1]
            else:
                a_i_1 = X
            da_i = da_i_1
            #print(f"da_{i}: {da_i}")
            dw_i, db_i, da_i_1 = self.layers[j].backward(da_i, a_i_1)
            #print(f"dw_i: {dw_i}, db_i: {db_i}, da_i_1: {da_i_1}")
            self.layers[j].weights -= lr * dw_i
            #print(f"Initial db_i:\n{db_i}")
            db_i = db_i.mean(axis=0)
            #print(f"db_i after taking mean:\n{db_i}")
            #print(f"Bias: {self.layers[j].bias.shape}, db_i: {db_i}")
            self.layers[j].bias -= lr * db_i
        

model = Model([3, 3, 2, 1])

preds = model.forward(data)
loss = cost(preds, labels)

print(f"Cost_1: {loss}")

for i in range(10):
    model.optimise(data, labels, 0.1)

preds = model.forward(data)
loss = cost(preds, labels)
print(f"Loss: {loss}")

def working(n_iters=10):
    wins = []
    for i in range(n_iters):
        model = Model([3, 3, 2, 1])
        loss1 = cost(model.forward(data), labels)
        for j in range(500):
            model.optimise(data, labels, 0.01)
        loss2 = cost(model.forward(data), labels)
        #if loss1 > loss2:
        #    wins.append(1)
        #else:
        #    wins.append(0)
        wins.append(loss2)

    wins = np.array(wins)

    print(wins)
    percentage_win = wins.mean(dtype=np.float64)
    print(f"Working percentage: {percentage_win}%")

working(100)
