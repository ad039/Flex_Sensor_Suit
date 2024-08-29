import matplotlib.pyplot as plt
import numpy as np
import pandas



# Prepare arrays x, y, z
data_csv = pandas.read_csv('PyTorch/data/Mediapipe_Pose_Estimation_Data_3.csv')

# mdeiapipe axes:
# x - horizontal right is +
# y - vertical, down is +
# z - into screen is +
ax = plt.figure().add_subplot(projection='3d')
ax.plot(data_csv.P_x, data_csv.P_y, data_csv.P_z, label='parametric curve')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()

fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(data_csv.ElbowFlex)
ax2.plot(data_csv.ShoulderFlex1)
ax2.plot(data_csv.ShoulderFlex2)
ax2.plot(data_csv.ShoulderFlex3)
ax2.plot(data_csv.ForearmFlex)
ax2.plot(data_csv.HandFlex1)
ax2.plot(data_csv.HandFlex2)
ax2.legend(["ElbowFlex", "ShoulderFlex1","ShoulderFlex2", "ShoulderFlex3", "ForearmFlex", "HandFlex1", "HandFlex2"])

fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(data_csv.P_x)
ax3.plot(data_csv.P_y)
ax3.plot(data_csv.P_z)
ax3.legend(["P_x", "P_y", "P_z"])

plt.show()