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
marker_size = 0.0785
 
# Calibration parameters yaml file
camera_calibration_parameters_filename_logi = 'Pose_Estimation/Camera_Calibration/calibration_chessboard_logi.yaml'
camera_calibration_parameters_filename_webc = 'Pose_Estimation/Camera_Calibration/calibration_chessboard_webc.yaml'

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

    vecs = []
    xyz = []
    rpy = []
        
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
       
        # Store the translation (i.e. position) information
        vecs.append(np.squeeze(tvecs[i][0][0]))     # x translation
        vecs.append(np.squeeze(tvecs[i][0][1]))     # y translation
        vecs.append(np.squeeze(tvecs[i][0][2]))     # z translation

 
        # Store the rotation information
        rotation_matrix = np.eye(4)
        rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
        r = R.from_matrix(rotation_matrix[0:3, 0:3])
        quat = r.as_quat()   
         
        # Quaternion format     
        transform_rotation_x = quat[0] 
        transform_rotation_y = quat[1] 
        transform_rotation_z = quat[2] 
        transform_rotation_w = quat[3] 
         
        # Euler angle format in radians
        roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                       transform_rotation_y, 
                                                       transform_rotation_z, 
                                                       transform_rotation_w)
        
        rotation_matrix = cv2.Rodrigues(np.array(rvecs[i][0]))[0]  #
        R_T = rotation_matrix.T
        T = tvecs[0].T

        xyz = np.squeeze(np.dot(R_T, - T))
        
        rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])

        vecs.append(math.degrees(roll_x))
        vecs.append(math.degrees(pitch_y))
        vecs.append(math.degrees(yaw_z))
         
        # Draw the axes on the marker
        cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvecs[i], tvecs[i], 0.05)
    return frame, xyz, rpy, np.array(vecs)
 
def main():
  """
  Main method of the program.
  """
 
  # Load the camera parameters from the saved file
  cv_file_logi = cv2.FileStorage(
    camera_calibration_parameters_filename_logi, cv2.FILE_STORAGE_READ) 
  mtx_logi = cv_file_logi.getNode('K').mat()
  dst_logi = cv_file_logi.getNode('D').mat()
  cv_file_logi.release()

  cv_file_webc = cv2.FileStorage(
  camera_calibration_parameters_filename_webc, cv2.FILE_STORAGE_READ) 
  mtx_webc = cv_file_webc.getNode('K').mat()
  dst_webc = cv_file_webc.getNode('D').mat()
  cv_file_webc.release()

  marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, -marker_size / 2, 0],
                        [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
   
  # Start the video stream
  cap1 = cv2.VideoCapture(0)
  cap2 = cv2.VideoCapture(2)
   
  while (cap1.isOpened() and cap2.isOpened()):
  
    # Capture frame-by-frame
    # This method returns True/False as well
    # as the video frame.
    ret, frame_logi = cap1.read() 
    ret, frame_webc = cap2.read()  

    frame_logi, xyz_logi, rpy_logi, vecs_logi = aruco_pose_estimation(frame_logi, mtx_logi, dst_logi)
    frame_webc, xyz_webc, rpy_webc, vecs_webc = aruco_pose_estimation(frame_webc, mtx_webc, dst_webc)

    # Display the resulting frame
    cv2.imshow('frame logi',frame_logi)
    cv2.imshow('frame webc',frame_webc)
    
    #print(vecs_logi, vecs_webc)  
    print(vecs_logi)
    '''
    theta_logi = np.deg2rad(-vecs_logi[3]+180)
    rotation_matrix_logi = np.array([[np.cos(theta_logi), 0, np.sin(theta_logi)],
                                     [0, 1, 0],
                                     [-np.sin(theta_logi), 0, np.cos(theta_logi)]])
    print((vecs_logi[0:3].reshape(3, -1)).shape)
    
    camera_logi_vec = np.matmul(rotation_matrix_logi, (vecs_logi[0:3].reshape(3, -1)))
    '''
    print(vecs_logi)
    print(xyz_logi, rpy_logi)

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