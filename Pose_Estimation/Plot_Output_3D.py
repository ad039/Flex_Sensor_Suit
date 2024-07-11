import matplotlib.pyplot as plt
import numpy as np
import pandas

ax = plt.figure().add_subplot(projection='3d')

# Prepare arrays x, y, z
data_csv = pandas.read_csv('Pose_Estimation/output_test_5min.csv', usecols=["P_x", "P_y", "P_z"])

print(data_csv.P_x)

# mdeiapipe axes:
# x - horizontal right is +
# y - vertical, down is +
# z - into screen is +
ax.plot(data_csv.P_x, data_csv.P_y, data_csv.P_z, label='parametric curve')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

plt.show()