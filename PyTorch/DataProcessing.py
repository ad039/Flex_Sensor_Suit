import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MotionCaptureData = pd.read_csv('./PyTorch/ALEX_MOTION_CAP_DATA_TRAIN_NEW(in).csv', header=[0,1,2], skiprows=1, skip_blank_lines=0,na_values=[''])
BLESensorData = pd.read_csv('./PyTorch/output_test_motion_cap.csv')

print(MotionCaptureData)
LABEL = MotionCaptureData.columns.tolist()[6]
print(LABEL)
ElbowY = MotionCaptureData.loc[:,('Arm:Elbow', 'Position', 'Y')].to_numpy()
shoulderFlex2 = BLESensorData.ShoulderFlex2.to_numpy()

fig1 = plt.figure()
ax1 = plt.axes()
ax1.plot(ElbowY[:2000])
ax1.plot(shoulderFlex2[:2000])
ax1.legend(["Motion Cap Arm:Elbow Y", "ShoulderFlex2"])
#plt.show()
points = plt.ginput(2)

dif = points[0][0]-points[1][0]
print(dif)
if dif >= 0:
    print("dif >= 0")
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(ElbowY[round(dif):])
    ax2.plot(shoulderFlex2)


    MotionCaptureData_new = MotionCaptureData.drop(MotionCaptureData.index[0:round(dif)], axis=0)
    print(MotionCaptureData_new)
    MotionCaptureData_new = MotionCaptureData_new.reset_index(drop=True)
    BLESensorData_new = BLESensorData
    print(MotionCaptureData_new)

else:
    dif = -dif
    fig2 = plt.figure()
    ax2 = plt.axes()
    ax2.plot(ElbowY)
    ax2.plot(shoulderFlex2[round(dif):])


    BLESensorData_new = BLESensorData.drop(BLESensorData.index[0:round(dif)], axis=0)
    print(BLESensorData_new)
    BLESensorData_new = BLESensorData_new.reset_index(drop=True)
    MotionCaptureData_new = MotionCaptureData
    print(BLESensorData_new)

fig3 = plt.figure()
ax3 = plt.axes()
ax3.plot(MotionCaptureData_new.loc[:, ('Arm:Elbow', 'Position', 'Y')])
ax3.plot(MotionCaptureData_new.loc[:, ('Hand', 'Position', 'Y')])
ax3.plot(BLESensorData_new.ShoulderFlex2)
plt.show()

Motion_Cap_Formatted = pd.concat([MotionCaptureData_new, BLESensorData_new], axis=1)

Motion_Cap_Formatted = Motion_Cap_Formatted.dropna()


Motion_Cap_Formatted.to_csv('./PyTorch/Alex_Motion_Cap_Test_Formatted_New.csv')



