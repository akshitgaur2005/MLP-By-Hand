from backend import Model, mse
import numpy as np

X = np.array([[1, 0],
             [0, 1]])
y = np.array([[1],
              [0]])

model = Model([2, 3, 1])
print(mse(model.forward(X), y))
model.fit(X, y, 500, 0.1)
print(mse(model.forward(X), y))

