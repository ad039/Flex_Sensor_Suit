import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


Data = np.loadtxt('./PyTorch/data/Alex_Motion_Cap_Test_Formatted_New.csv', skiprows=1, delimiter=",", dtype=np.float32)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()


def RMSE(y_pred, y):
    return np.sqrt(np.square(np.subtract(y_pred,y)).mean())

target = Data[:, 15]
target = target.reshape(-1,1)
target = scaler_y.fit_transform(target)

sensor = Data[:, [38, 39, 40]]
sensor = scaler_x.fit_transform(sensor)

num_epochs = 10000
prevloss = 10000

for i  in range(num_epochs):
    weights1 = np.random.rand(3,3)
    weights2 = np.random.rand(3,1)

    result = (sensor.dot(weights1)).dot(weights2)
    loss = RMSE(result, target)

    if loss < prevloss:
        prevloss = loss
        weights1_best = weights1
        weights2_best = weights2


test = scaler_y.inverse_transform((sensor.dot(weights1_best)).dot(weights2_best))
target = scaler_y.inverse_transform(target)
print(RMSE(test, target))

fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(target)
ax3.plot(test)
ax3.legend(["Target", "Sensor"])
plt.show()