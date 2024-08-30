import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

dataset = np.loadtxt('./PyTorch/data/Optitrack_Mocap_Data.csv', delimiter=",", dtype=np.float32, skiprows=1)

hand_centre = dataset[:, 12]
shoulder_centre = dataset[:, 6]
hand_shoulder_origin = np.subtract(hand_centre, shoulder_centre)
#hand_shoulder_origin = np.around(hand_shoulder_origin/5, decimals=0)*5 # round to the nearest 5 for training



# Load the FSS dataset
FSS_X = dataset[:, 37:44]
FSS_y = hand_shoulder_origin

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
print(diabetes_X.shape, diabetes_y.shape)
print(FSS_X.shape, FSS_y.shape)

# Use only one feature
FSS_X = FSS_X[:, np.newaxis, 2]

# Split the data into training/testing sets
FSS_X_train = FSS_X[:-20]
FSS_X_test = FSS_X[-20:]

# Split the targets into training/testing sets
FSS_y_train = FSS_y[:-20]
FSS_y_test = FSS_y[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(FSS_X_train, FSS_y_train)

# Make predictions using the testing set
FSS_y_pred = regr.predict(FSS_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(FSS_y_test, FSS_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(FSS_y_test, FSS_y_pred))

# Plot outputs
plt.scatter(FSS_X_test, FSS_y_test, color="black")
plt.plot(FSS_X_test, FSS_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()