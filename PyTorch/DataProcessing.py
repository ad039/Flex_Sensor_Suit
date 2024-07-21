import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MotionCaptureData = pd.read_csv('./PyTorch/Alex_Motion_Cap_1_edited.csv', skiprows=2, skip_blank_lines=0,na_values=[''])
BLESensorData = pd.read_csv('./PyTorch/output_test_motion_cap.csv')

plt.plot(MotionCaptureData.loc[:,"Arm:Hand TopY"])
plt.plot(BLESensorData.ShoulderFlex1)
plt.legend(["Motion Cap", "ShoulderFlex2"])
plt.show()

#.to_csv('./PyTorch/Alex_Motion_Cap_Formatted.csv')



