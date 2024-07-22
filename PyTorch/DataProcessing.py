import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MotionCaptureData = pd.read_csv('./PyTorch/Alex_Motion_Cap_1_edited.csv', skiprows=2, skip_blank_lines=0,na_values=[''])
BLESensorData = pd.read_csv('./PyTorch/output_test_motion_cap.csv')

ElbowY = MotionCaptureData.loc[:,"Arm:ElbowY"].to_numpy()
shoulderFlex2 = BLESensorData.ShoulderFlex2.to_numpy()

fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(ElbowY[:500])
ax1.plot(shoulderFlex2[:500])
ax1.legend(["Motion Cap Arm:ElbowY", "ShoulderFlex2"])
#plt.show()
points = plt.ginput(2)

dif = points[0][0]-points[1][0]
print(dif)

fig2 = plt.figure()
ax2 = plt.axes()
ax2.plot(ElbowY[round(dif):])
ax2.plot(shoulderFlex2)


MotionCaptureData_new = MotionCaptureData.drop(MotionCaptureData.index[0:round(dif)], axis=0)
print(MotionCaptureData_new.head())
MotionCaptureData_new = MotionCaptureData_new.reset_index(drop=True)
print(MotionCaptureData_new.head())
fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(MotionCaptureData_new.loc[:, "Arm:ElbowY"])
ax3.plot(MotionCaptureData_new.loc[:, "Arm:Hand TopY"])
ax3.plot(BLESensorData.ShoulderFlex2)
plt.show()

Motion_Cap_Formatted = pd.concat([MotionCaptureData_new, BLESensorData], axis=1)

Motion_Cap_Formatted = Motion_Cap_Formatted.dropna()


Motion_Cap_Formatted.to_csv('./PyTorch/Alex_Motion_Cap_Formatted.csv')



