import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.animation as animation



# read BLE
from bluepy import btle
import time
import datetime
import csv
from threading import Thread, Event
import queue
import struct
import numpy as np


# function to communicate with microcontroller and recieve sensor data over bluetooth
def ble_task(queue=queue.LifoQueue):
    global ready

    print("Connecting...", end='\r')

    # MAC addresses for ble chips
    # "29:F0:E3:F9:C9:CD" for Arduino Nano BLE
    # "93:43:92:07:91:11" for Xioa nrf52 BLE 
    # "F4:12:FA:5A:39:51" for esp32-s3 qt py


    FlexSensorSuit = btle.Peripheral("93:43:92:07:91:11")

    print("connected", end='\r')
    FlexSensorSuit.getServices()
    flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
    print("connected to service", end='\r')

    flexSensorCharValue = flexSensorService.getCharacteristics()[0]
    print("connected to characteristic")
    i = 0

    while True:
        # wait for writer to be ready
        writer_ready_evnt.wait()

        val = flexSensorCharValue.read()
        val = struct.unpack("<hhhhhhh",val)
        #print(val)
        queue.put(val)


        if (train_over_evnt.is_set()):
                break
        
        if (test_over_evnt.is_set()):
                break
        
    FlexSensorSuit.disconnect()
    return

import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# function to run pose detection using mediapipe
def pose_estimation(queue=queue.LifoQueue):

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(2)

    time_prev = 0
    i = 0
    
    with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9, model_complexity=2) as pose:
        
        while cap.isOpened():
            
            # wait for writer to be ready
            writer_ready_evnt.wait()

            success, image = cap.read()

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB) # flip to get selfi view, this impacts the landmark extraction later on: take left hand for right hand

            image.flags.writeable = False
            
            results = pose.process(image)

            image.flags.writeable = True
            image_height, image_width, image_depth = image.shape

            if results.pose_landmarks is not None:
                right_shoulder = [results.pose_world_landmarks.landmark[11].x,results.pose_world_landmarks.landmark[11].y, results.pose_world_landmarks.landmark[11].z]
                right_wrist = [results.pose_world_landmarks.landmark[15].x,results.pose_world_landmarks.landmark[15].y, results.pose_world_landmarks.landmark[15].z]
                right_index = [results.pose_world_landmarks.landmark[19].x,results.pose_world_landmarks.landmark[19].y, results.pose_world_landmarks.landmark[19].z]
                #right_pinky = [results.pose_world_landmarks.landmark[17].x,results.pose_world_landmarks.landmark[17].y, results.pose_world_landmarks.landmark[17].z]
                
                hand_centre_world = [np.mean([right_wrist, right_index], axis=0)]

                right_hand_norm = np.subtract(hand_centre_world, right_shoulder)


                queue.put(right_hand_norm)
                
                
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Mediapipe pose', image)

            if cv2.waitKey(1) & 0xFF ==27:
                break
                
            if (train_over_evnt.is_set()):
                break

    cap.release()
    cv2.destroyAllWindows()
    return


# function to control reading the queues and sending that data to a csv file (output.csv) for training
def writer_task_train(ble_queue=queue.Queue, pose_queue=queue.Queue, writer=None):
    prevTime = 0
    startTime = time.perf_counter()

    while(True):
        # set writer ready event
        writer_ready_evnt.set()

        print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}', end='\r')
        prevTime = time.perf_counter()
        # read the queues from BLE and Pose Estimation
        pose_val = pose_queue.get()
        ble_val = ble_queue.get()

        # close ready event
        writer_ready_evnt.clear()

        # empty the queues so they dont clog up
        with pose_queue.mutex:
            pose_queue.queue.clear()
        
        with ble_queue.mutex:
            ble_queue.queue.clear()

        # append all data into a numpy array
        queue_val = np.append(pose_val, np.divide(ble_val, 100))
        #print(queue_val)
        writer.writerow(queue_val)
        #print(f'duration: {(time.perf_counter()-prevTime)*1000:.3f}')

        if (time.perf_counter() - startTime > 2*60):
            train_over_evnt.set()
            break
    return

# function to control reading the queues and sending that data to a csv file (output_test.csv) for testing
def writer_task_test(ble_queue=queue.Queue, writer=None):
    prevTime = 0
    startTime = time.perf_counter()

    while(True):
        # set writer ready event
        writer_ready_evnt.set()

        print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}', end='\r')
        prevTime = time.perf_counter()
        # read the queues from BLE and Pose Estimation
        ble_val = ble_queue.get()

        # close ready event
        writer_ready_evnt.clear()

        # empty the queues so they dont clog up
        with ble_queue.mutex:
            ble_queue.queue.clear()

        # append all data into a numpy array
        queue_val = np.divide(ble_val, 100).reshape(1, -1)

        queue_val_scaled = scaler.transform(queue_val)

        # run the queue through the RFR models
        x_pred = model_x.predict(queue_val_scaled)
        y_pred = model_y.predict(queue_val_scaled)
        z_pred = model_z.predict(queue_val_scaled)

        # put all values into an array
        predictions = np.array([x_pred, y_pred, z_pred])
        writer_val = np.append(predictions, queue_val)

        # write to a file
        #print(writer_val)
        writer.writerow(writer_val)
        #print(f'duration: {(time.perf_counter()-prevTime)*1000:.3f}')


        if (time.perf_counter() - startTime > 0.3*60):
            test_over_evnt.set()
            break
    return

# function to train the random forest regressor network
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def rfr_train(model_x=RandomForestRegressor, model_y=RandomForestRegressor, model_z=RandomForestRegressor, scaler=StandardScaler):

    # read the data from the file
    columns = ["target_x", "target_y", "target_z",
               "elbow", "shoulder_1", "shoulder_2", 
               "shoulder_3", "forearm", "hand_1", "hand_2"]
    
    data = pd.read_csv('./output.csv', header=None, names=columns)

    X_1 = data[['elbow', 'shoulder_1', 'shoulder_2', 
               'shoulder_3', 'forearm', 'hand_1', 'hand_2']]
    
    y_1 = data["target_x"]*1000
    y_2 = data["target_y"]*1000
    y_3 = data["target_z"]*1000 # convert to mm

    X_1 = scaler.fit_transform(X_1.to_numpy())

    X1_train, X1_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(X_1, y_1, y_2, y_3, test_size=0.1, random_state=42)

    y1_train_np = y1_train.values.ravel()
    y2_train_np = y2_train.values.ravel()
    y3_train_np = y3_train.values.ravel()


    model_x.fit(X1_train, y1_train_np)
    model_y.fit(X1_train, y2_train_np)
    model_z.fit(X1_train, y3_train_np)

    # test
    y1_pred = model_x.predict(X1_test)
    y2_pred = model_y.predict(X1_test)
    y3_pred = model_z.predict(X1_test)

    # convert to np arrays
    y1_test_np = np.array(y1_test)
    y2_test_np = np.array(y2_test)
    y3_test_np = np.array(y3_test)

    y1_pred_np = np.array(y1_pred)
    y2_pred_np = np.array(y2_pred)
    y3_pred_np = np.array(y3_pred)

    # RMSE
    rmse_x = np.sqrt(mean_squared_error(y1_test_np, y1_pred_np))
    rmse_y = np.sqrt(mean_squared_error(y2_test_np, y2_pred_np))
    rmse_z = np.sqrt(mean_squared_error(y3_test_np, y3_pred_np))

    print("rmse_x = ", rmse_x, "rmse_y = ", rmse_y, "rmse_z = ", rmse_z)



# main function
if __name__ == "__main__":

    # two Lifo queues to get the most recent data so data alignes better
    ble_q = queue.LifoQueue(maxsize=100)
    pose_q = queue.LifoQueue(maxsize=100)

    # events
    writer_ready_evnt = Event()
    train_over_evnt = Event()
    test_over_evnt = Event()

    with open('./output.csv', 'w', newline='') as f:

        print("Collecting Training Data...\n")

        # clear all events
        train_over_evnt.clear()
        writer_ready_evnt.clear()
        test_over_evnt.clear()

        writer = csv.writer(f)

        ble_thread = Thread(target=ble_task, args=(ble_q,))
        pose_estimation_thread = Thread(target=pose_estimation, args=(pose_q,))
        writer_thread = Thread(target=writer_task_train, args=(ble_q, pose_q, writer))
        
        ble_thread.start()
        pose_estimation_thread.start()

        time.sleep(5)
        writer_thread.start()

        ble_thread.join()
        pose_estimation_thread.join()
        writer_thread.join()

        f.close()
    
    print("\nTraining RFR Networks...\n")

    # initailise RFR models
    model_x = RandomForestRegressor(n_estimators=500, random_state=42)
    model_y = RandomForestRegressor(n_estimators=500, random_state=42)
    model_z = RandomForestRegressor(n_estimators=500, random_state=42)

    scaler = StandardScaler()

    # with this new data train a random forest regression network
    rfr_train(model_x=model_x, model_y=model_y, model_z=model_z, scaler=scaler)

    # begin prediction of hand position
    with open('./output_test.csv', 'w', newline='') as f:

        print("\nTesting...\n")

        # clear all events
        train_over_evnt.clear()
        writer_ready_evnt.clear()
        test_over_evnt.clear()
        
        writer = csv.writer(f)

        ble_thread = Thread(target=ble_task, args=(ble_q,))

        ble_thread.start()

        time.sleep(5)
        writer_task_test(ble_q, writer)

        
        ble_thread.join()

        print("\nDone Testing, Plotting...\n")

        # plot test data
        # read the data from the file
        columns = ["target_x", "target_y", "target_z",
                "elbow", "shoulder_1", "shoulder_2", 
                "shoulder_3", "forearm", "hand_1", "hand_2"]
        
        data = pd.read_csv('./output_test.csv', header=None, names=columns)
        
        x = data["target_x"]
        y = data["target_y"]
        z = data["target_z"]

        fig = plt.figure()
        ax = plt.axes(projection='3d')  
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

        ax.plot3D(x, y, z)
        plt.show()


        f.close()



    
