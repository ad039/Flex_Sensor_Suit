import cv2
import time

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(4)

count = 0

time.sleep(10)
prevTime = time.perf_counter()

while (cap1.isOpened()):# and cap2.isOpened()):
        
    success1, image1 = cap1.read()
    success2, image2 = cap2.read()
    
    cv2.imshow('Cam 1', image1)
    cv2.imshow('Cam 2', image2)

    
    if (time.perf_counter() - prevTime) > 5:
        prevTime = time.perf_counter()

        print("capture ", count)

        name1 = "Pose_Estimation/Camera_Calibration/Logitech_1/%d.jpg"%count
        name2 = "Pose_Estimation/Camera_Calibration/Logitech_2/%d.jpg"%count
        
        cv2.imwrite(name1, image1)     # save frame as JPEG file
        cv2.imwrite(name2, image2)     # save frame as JPEG file
        
        count += 1
    

    if cv2.waitKey(1) & 0xFF ==27:
        break

cap1.release()
cap2.release()