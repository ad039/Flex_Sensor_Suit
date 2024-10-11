#!/usr/bin/env python
  
'''
Welcome to the ArUco Marker Pose Estimator!
  
This program:
  - Estimates the pose of an ArUco Marker
'''
  
from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
from scipy.spatial.transform import Rotation as R
import math # Math library
 
# Project: ArUco Marker Pose Estimator
# Date created: 12/21/2021
# Python version: 3.8
 
# Dictionary that was used to generate the ArUco marker 
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
 
# Side length of the ArUco marker in meters 
marker_size = 0.203
 
# Calibration parameters yaml file
camera_calibration_parameters_filename_logi_1 = 'Pose_Estimation/Camera_Calibration/calibration_chessboard_logi_2.yaml'
camera_calibration_parameters_filename_logi_2 = 'Pose_Estimation/Camera_Calibration/calibration_chessboard_logi_1.yaml'

def euler_from_quaternion(x, y, z, w):
  """
  Convert a quaternion into euler angles (roll, pitch, yaw)
  roll is rotation around x in radians (counterclockwise)
  pitch is rotation around y in radians (counterclockwise)
  yaw is rotation around z in radians (counterclockwise)
  """
  t0 = +2.0 * (w * x + y * z)
  t1 = +1.0 - 2.0 * (x * x + y * y)
  roll_x = math.atan2(t0, t1)
      
  t2 = +2.0 * (w * y - z * x)
  t2 = +1.0 if t2 > +1.0 else t2
  t2 = -1.0 if t2 < -1.0 else t2
  pitch_y = math.asin(t2)
      
  t3 = +2.0 * (w * z + x * y)
  t4 = +1.0 - 2.0 * (y * y + z * z)
  yaw_z = math.atan2(t3, t4)
      
  return roll_x, pitch_y, yaw_z # in radians

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return np.array([rvecs]), np.array([tvecs]), trash

def aruco_pose_estimation(frame, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250) # cv2.aruco.DICT_4X4_250 seleccion del modelo Aruco
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    transformation_matrix = np.zeros([4, 4])
        
    # Detect ArUco markers in the video frame
    (corners, marker_ids, rejected) = detector.detectMarkers(
      frame)
       
    # Check that at least one ArUco marker was detected
    if marker_ids is not None:
 
      # Draw a square around detected markers in the video frame
      cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)

      # Get the rotation and translation vectors
      rvecs, tvecs, obj_points = my_estimatePoseSingleMarkers(
        corners,
        marker_size,
        matrix_coefficients,
        distortion_coefficients)
      
      # Print the pose for the ArUco marker
      # The pose of the marker is with respect to the camera lens frame.
      # Imagine you are looking through the camera viewfinder, 
      # the camera lens frame's:
      # x-axis points to the right
      # y-axis points straight down towards your toes
      # z-axis points straight ahead away from your eye, out of the camera
      for i, marker_id in enumerate(marker_ids):
        

        
        rotation_matrix = cv2.Rodrigues(np.array(rvecs[i][0]))[0]  #
        #print("rotation matrix = ", rotation_matrix)
        translation_vector = tvecs[0]
        #print("translation vector = ", translation_vector)
        
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0:3, [3]] = translation_vector
        transformation_matrix[3, 3] = 1

         
        # Draw the axes on the marker
        cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvecs[i], tvecs[i], 0.05)
    return frame, transformation_matrix
 
def main():
  """
  Main method of the program.
  """
 
  # Load the camera parameters from the saved file
  cv_file_logi_1 = cv2.FileStorage(
    camera_calibration_parameters_filename_logi_1, cv2.FILE_STORAGE_READ) 
  mtx_1 = cv_file_logi_1.getNode('K').mat()
  dst_1 = cv_file_logi_1.getNode('D').mat()
  cv_file_logi_1.release()

  cv_file_logi_2 = cv2.FileStorage(
  camera_calibration_parameters_filename_logi_2, cv2.FILE_STORAGE_READ) 
  mtx_2 = cv_file_logi_2.getNode('K').mat()
  dst_2 = cv_file_logi_2.getNode('D').mat()
  cv_file_logi_2.release()

  marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, -marker_size / 2, 0],
                        [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
   
  # Start the video stream
  cap1 = cv2.VideoCapture(0)
  cap2 = cv2.VideoCapture(2)

  cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
   
  while (cap1.isOpened() and cap2.isOpened()):
  
    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame_1 = cap1.read() 
    ret, frame_2 = cap2.read()  

    frame_1, T_logi1_marker = aruco_pose_estimation(frame_1, mtx_1, dst_1)
    frame_2, T_logi2_marker = aruco_pose_estimation(frame_2, mtx_2, dst_2)

    # ROTATE AND RESIZE
    frame_1 = cv2.rotate(frame_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_2 = cv2.rotate(frame_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    frame_1 = cv2.resize(frame_1, (340, 600))
    frame_2 = cv2.resize(frame_2, (340, 600))

    # Display the resulting frame
    cv2.imshow('frame 1', frame_1)
    cv2.imshow('frame 2', frame_2)

    #print("T_logi1_marker = ", T_logi1_marker)
    #print("T_logi2_marker = ", T_logi2_marker)

    if (np.linalg.det(T_logi1_marker) != 0.0) and (np.linalg.det(T_logi2_marker) != 0.0):

      T_marker_logi1 = np.linalg.inv(T_logi1_marker)
      T_marker_logi2 = np.linalg.inv(T_logi2_marker)
      
      #print("T_marker_logi1 = ", T_marker_logi1)
      #print("T_marker_logi2 = ",T_marker_logi2)

      T_logi1_logi2 = T_logi1_marker @ T_marker_logi2

      dist = np.linalg.norm(T_logi2_marker[0:3, 3])

      print(dist)



    # If "q" is pressed on the keyboard, 
    # exit this loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  # Close down the video stream
  cap1.release()
  cap2.release()
  cv2.destroyAllWindows()
   
if __name__ == '__main__':
    print(__doc__)
    main()