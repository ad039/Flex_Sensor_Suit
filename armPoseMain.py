#Arm Pose Main

import numpy as np
import cv2
import mediapipe as mp
import time




def pose_estimation():
    global mp_drawing, mp_pose, pose

	# initialize Pose estimator
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)
    mp.solutions.num_faces = 1
    


def main():
    # Call all functions
    pose_estimation()

    # For FPS
    prev_frame_time = 0
    new_frame_time = 0
    prevLoopTime = 0
	
    # define a video capture location
    vid = cv2.VideoCapture(0)

    try:
        while True:
            if time.perf_counter() - prevLoopTime > 0.00:
                prevLoopTime = time.perf_counter()
                success, color_image = vid.read()        
                # ----------------------------  POSE ESTIMATION ----------------------------#
                results = pose.process(color_image)
                #print(results.pose_landmarks)

                # Extract landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                    # Get coordinates
                    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z]
                    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                    
                    right_wrist_norm = np.subtract(right_wrist,right_shoulder)
                    print(right_wrist_norm)
                except:
                    pass

                # draw dected skeleton on the frame
                mp_drawing.draw_landmarks(color_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # FPS counter
                new_frame_time = time.time()	
                fps = 1/(new_frame_time - prev_frame_time)
                prev_frame_time = new_frame_time
                fps = int(fps)
                fps = str(fps)
                cv2.putText(color_image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            
                # Show images
                cv2.namedWindow('Pose Estimation', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Pose Estimation', color_image)	# Only show color image (not depth)
                cv2.waitKey(1)
            
    finally:
	    # Stop streaming
        vid.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
	main()