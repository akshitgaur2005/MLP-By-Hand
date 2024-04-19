import numpy as np

def relu(x):
    """
    ReLU - Rectified Linear Unit
    returns y = x if x > 0
              = 0 otherwise
    """
    return x * (x > 0)

def drelu(x):
    """
    Derivative of ReLU
    returns y = 1 if x > 0
              = 0 otherwise
    """
    return 1 * (x > 0)

def mse(y_hat, y):
    """
    Mean Squared Error (divided by 2 as it makes its derivative cleaner)
    """
    return (np.mean(np.square(y_hat - y)) / 2)

class Dense():

    def __init__(self, in_feats, out_feats):
        """
        A simple fully connected layer
            Each output from previous layer is passed onto each neuron in the
            layer as input.
        """
        self.weights = np.random.randn(in_feats, out_feats)
        self.bias  = np.random.randn(1, out_feats)

    def forward(self, X):
        """
        Forward pass
        returns
            X * W + b
        """
        return np.dot(X, self.weights) + self.bias

    def backward(self, da_i, a_i_1):
        """
        It generates three terms responsible for performing backpropagation
        inputs are
            da_i = dJ / da_i = Derivative of Cost function with respect to activation of current layer
            a_i_1 = a_{i-1} = Input provided to the layer
        returns
            dw_i = dJ / dw_i = Gradient of the weights of current layer to the Cost function
            db_i = dJ / db_i = Gradient of the bias of the current layer to the Cost function
            da_i_1 = dJ / da_{i-1} = Gradient of the activation of previous layer to the Cost function
        """
        dw_i = np.dot(a_i_1.T, np.multiply(da_i, drelu(self.forward(a_i_1))))
        db_i = np.multiply(da_i, drelu(self.forward(a_i_1)))
        da_i_1 = np.dot(np.multiply(da_i, relu(self.forward(a_i_1))), self.weights.T)
        return dw_i, db_i, da_i_1

class Model():

    def __init__(self, neurons=[3, 3, 2, 1]):
        """
        Model
        inouts are
            neurons = array of no. of neurons in layers (first layer is the no. of input features)
        """
        self.layers = []
        for i in range(len(neurons) - 1):
            self.layers.append(Dense(neurons[i], neurons[i + 1]))

    def forward(self, X, train=False):
        """
        Forward Pass
        inputs are
            X = input data
            train = to track outputs of each layer or not
        returns
            prediction if train=False
            array of output of each value
        """
        input = X
        if train:
            outputs = []
        for i in range(len(self.layers) - 1):
            input = relu(self.layers[i].forward(input))
            if train:
                outputs.append(input)
        preds = self.layers[-1].forward(input)
        if train:
            outputs.append(preds)
            return outputs
        return preds

    def optimise(self, X, y, lr):
        """
        Performs a single step of optimisation
        """
        outputs = self.forward(X, train=True)
        a_L = outputs[-1]
        da_i_1 = (a_L - y) / y.shape[1]

        for i in range(len(self.layers)):
            j = len(self.layers) - 1 - i
            if j > 0:
                a_i_1 = outputs[j - 1]
            else:
                a_i_1 = X
            da_i = da_i_1
            dw_i, db_i, da_i_1 = self.layers[j].backward(da_i, a_i_1)
            db_i = np.mean(db_i, axis=0)
            self.layers[j].weights -= lr * dw_i
            self.layers[j].bias -= lr * db_i

    def fit(self, X, y, epochs, lr):
        """
        Fits the model to a dataset
        inputs are
            X = features
            y = labels
        returns
            array tracking loss across epochs
        """
        loss = []
        #track_var = np.round(epochs / 10)
        track_var = 1
        for i in range(epochs):
            self.optimise(X, y, lr)
            if (i % track_var == 0):
                loss.append(mse(self.forward(X), y))
        return loss
