import cv2

# dictionary to specify type of the marker
marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

# MARKER_ID = 0
MARKER_SIZE = 400  # pixels

# generating unique IDs using for loop
for id in range(20):  # genereting 20 markers
    # using funtion to draw a marker
    marker_image = cv2.aruco.generateImageMarker(marker_dict, id, MARKER_SIZE)
    cv2.imshow("img", marker_image)
    cv2.imwrite(f"Pose_Estimation/markers/marker_{id}.png", marker_image)
    if cv2.waitKey(500) & 0xFF ==27:
        break