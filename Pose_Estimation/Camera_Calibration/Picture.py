import cv2
import time

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

count = 0

time.sleep(10)
prevTime = time.perf_counter()

while (cap1.isOpened() and cap2.isOpened()):
        
    success1, image1 = cap1.read()
    success2, image2 = cap2.read()

    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    # Draw the pose annotations on the image.
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Logitech', image1)
    cv2.imshow('Webcam', image2)

    if (time.perf_counter() - prevTime) > 5:
        prevTime = time.perf_counter()

        name1 = "Pose_Estimation/Camera_Calibration/Logitech/%d.jpg"%count
        name2 = "Pose_Estimation/Camera_Calibration/Webcam/%d.jpg"%count
        
        cv2.imwrite(name1, image1)     # save frame as JPEG file
        cv2.imwrite(name2, image2)     # save frame as JPEG file
        
        count += 1


    if cv2.waitKey(1) & 0xFF ==27:
        break

cap1.release()
cap2.release()