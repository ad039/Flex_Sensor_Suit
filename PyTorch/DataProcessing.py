import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MotionCaptureData = pd.read_csv('./PyTorch/Alex_Motion_Cap_1_edited.csv', skiprows=2, skip_blank_lines=0,na_values=[''])
BLESensorData = pd.read_csv('./PyTorch/output_test_motion_cap.csv')

ElbowY = MotionCaptureData.loc[:,"Arm:Hand TopY"].to_numpy()
shoulderFlex1 = BLESensorData.ShoulderFlex1.to_numpy()

plt.figure()
plt.plot(ElbowY)
plt.plot(shoulderFlex1)
plt.legend(["Motion Cap Arm:Hand TopY", "ShoulderFlex2"])
#plt.show()
points = plt.ginput(2)

dif = points[0][0]-points[1][0]
print(dif)

plt.figure()
plt.plot(ElbowY[round(dif):])
plt.plot(shoulderFlex1)
plt.show()

#.to_csv('./PyTorch/Alex_Motion_Cap_Formatted.csv')



