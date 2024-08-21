from bluepy import btle
import time
import datetime
import csv
from queue import Queue
import threading
import struct
import numpy as np

class MyDelegate(btle.DefaultDelegate):
    def __init__(self):
        btle.DefaultDelegate.__init__(self)
        # ... initialise here

    def handleNotification(self, cHandle, data):
        # ... perhaps check cHandle
        # ... process 'data'
        ble_q.put(data)

def ble_task():
    # Initialisation  -------
    #"29:F0:E3:F9:C9:CD" for Arduino Nano BLE
    #"93:43:92:07:91:11" for Xioa nrf52 BLE 
    #"F4:12:FA:5A:39:51" for esp32-s3 qt py

    FlexSensorSuit = btle.Peripheral("F4:12:FA:5A:39:51")

    print("connected")
    FlexSensorSuit.getServices()
    flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
    print("connected to service")

    flexSensorChar = flexSensorService.getCharacteristics()[0]
    print("connected to characteristic")

    FlexSensorSuit.setDelegate(MyDelegate())

    FlexSensorSuit.writeCharacteristic(flexSensorChar.getHandle()+1, b"\x01\x00") # this starts the notification

    # Main loop --------
    while True:
        if FlexSensorSuit.waitForNotifications(1.0):
            # handleNotification() was called
            continue


def writer_task(ble_queue, f):
    prevTime_writer = 0.0
    
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "ElbowFlex", "ShoulderFlex1", "ShoulderFlex2", "ShoulderFlex3", "ForearmFlex", "HandFlex1", "HandFlex2"])

    while True:
        # wait for and read the queue from BLE_read
        ble_val = ble_queue.get()
        ble_val = struct.unpack("<hhhhhhh",ble_val) 
        writer_time = (time.perf_counter()-prevTime_writer)*1000
        prevTime_writer = time.perf_counter()
        writer.writerow(np.append(writer_time, np.divide(ble_val,100)))


if __name__ == "__main__":
    
    ble_q = Queue(maxsize=1)

    with open('Pose_Estimation/output_notify.csv', 'w', newline='') as f:
        ble_thread = threading.Thread(target=ble_task, args=())
        writer_thread = threading.Thread(target=writer_task, args=(ble_q, f))

        ble_thread.start()
        writer_thread.start()

        ble_thread.join()
        writer_thread.join()