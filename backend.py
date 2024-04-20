import numpy as np
import os

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

def clip(x, i):
    return np.clip(x, -i, i)

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

    def optimise(self, X, y, lr, counter):
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
            clip_val = 1e4
            dw_i = clip(dw_i, clip_val)
            db_i = clip(db_i, clip_val)
            if (counter % 100 == 0):
                print(f"Layer: {j} {self.layers[j].weights.shape}")
                print(f"lr: {lr}, dw_{j}: {np.mean(dw_i)}, lr * dw_{j}: {lr * np.mean(dw_i)}, W: {np.mean(self.layers[j].weights)}, b: {np.mean(self.layers[j].bias)}")
            ld = 0
            self.layers[j].weights -= lr * (dw_i + ld * self.layers[j].weights)
            self.layers[j].bias -= lr * db_i

        return a_L

    def fit(self, X, y, X_val, y_val, tolerance, epochs, lr, lr_decay, lr_min):
        """
        Fits the model to a dataset
        inputs are
            X = features
            y = labels
        returns
            array tracking loss across epochs
        """
        train_loss = []
        val_loss = []
        counter = 0
        train_mean = np.mean(y)
        val_mean = np.mean(y_val)
        
        #track_var = np.round(epochs / 10)
        #X = np.array_split(X, 10)
        #y = np.array_split(y, 10)
        for i in range(epochs):
            if (i % 100 == 0):
                os.system("clear")
            #pred = self.optimise(X[i % 10], y[i % 10], lr)
            pred = self.optimise(X, y, lr, i)
            train_loss.append(mse(pred, y))
            val_loss.append(mse(self.forward(X_val), y_val))
            if (i % 100 == 0):
                print(f"Epoch: {i}")
                print(f"pred: {pred[0]}, y: {y[0]}, diff: {np.sqrt(mse(pred, y)) / train_mean * 100}%, Val Loss: {np.sqrt(val_loss[i]) / val_mean * 100}%")
            if (i != 0):
                if (val_loss[i] >= val_loss[i - 1]):
                    counter += 1
                else:
                    counter = 0
            if counter == tolerance:
                break
            lr = max(lr * (lr_decay ** i), lr_min)
            
        return train_loss, val_loss
