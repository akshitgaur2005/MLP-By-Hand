from getdata import get_data
from backend import Model, mse
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()

X_train, y, ids, X_test = get_data()

X_train = X_train.to_numpy()
y = y.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()

#y_train = scaler.fit_transform(y)
y_train = y
model = Model([X_train.shape[1], 64, 32, 3, 1])
preds = model.forward(X_train)
print(preds[0])
#preds = scaler.inverse_transform(preds)
#print(f"Shapes:\nX_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, ids: {ids.shape}")
loss = mse(preds, y)
loss = model.fit(X_train, y_train, 20000, 1e-4, 0.99999, 1e-8)
#for i in range(1000):
    #print(f"Epoch: {i}")
    #model.optimise(X_train, y_train, lr)
    #lr = max(lr * 0.9, 1e-13)
new_preds = model.forward(X_train)
#new_preds = scaler.inverse_transform(new_preds)
y = y.reshape(-1).astype(np.float32)
new_preds = new_preds.reshape(-1)
new_loss = mse(new_preds, y)
i = np.random.randint(0, y.shape[0])
print(f"y:    {y[i]}")
print(f"Pred: {new_preds[i]}")
print(f"Diff: {y[i] - new_preds[i]}")
print(f"Loss: {loss[-1]}")
print(f"Loss:  {np.sqrt(new_loss)}")
