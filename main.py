from getdata import get_data
from backend import Model, mse
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

scaler = StandardScaler()

X_train, y_train, X_val, y_val, ids, X_test = get_data(0.33)

X_train = X_train.to_numpy()
y_train = y_train.to_numpy().reshape(-1, 1)
X_val = X_val.to_numpy()
y_val = y_val.to_numpy().reshape(-1, 1)
X_test = X_test.to_numpy()
ids = ids.to_numpy().reshape(-1)

#y_train = scaler.fit_transform(y)
model = Model([X_train.shape[1], 16, 1])
preds = model.forward(X_val)
print(preds[0])
#preds = scaler.inverse_transform(preds)
#print(f"Shapes:\nX_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, ids: {ids.shape}")
loss = mse(preds, y_val)
loss, val_loss = model.fit(X_train, y_train, X_val, y_val, 15, 10000, 1e-3, 0.9999, 1e-5)
#for i in range(1000):
    #print(f"Epoch: {i}")
    #model.optimise(X_train, y_train, lr)
    #lr = max(lr * 0.9, 1e-13)
new_preds = model.forward(X_val)
#new_preds = scaler.inverse_transform(new_preds)
y_train = y_train.reshape(-1).astype(np.float32)
new_preds = new_preds.reshape(-1)
new_loss = mse(new_preds, y_val)
i = np.random.randint(0, y_val.shape[0])
print(f"y:    {y_val[i]}")
print(f"Pred: {new_preds[i]}")
print(f"Diff: {y_val[i] - new_preds[i]}")
print(f"Loss:      {loss[-1]}")
print(f"Val Loss:  {val_loss[-1]}")

y_test = model.forward(X_test).reshape(-1)
print(ids.shape, y_test.shape)
res = [ids, y_test]
cols = ["Id", "SalePrice"]
results = pd.DataFrame({
    "Id": ids,
    "SalePrice": y_test
    })
results.to_csv("result.csv", index=False)

loss_pd = pd.DataFrame({
    "Train_Loss": loss,
    "Val_Loss": val_loss
    })

sns.lineplot(data=loss_pd)
plt.show()
