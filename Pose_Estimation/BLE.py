from bluepy import btle
import numpy as np
import struct
import time
import threading
from queue import Queue
import csv

#"29:F0:E3:F9:C9:CD" for Arduino Nano BLE
#"93:43:92:07:91:11" for Xioa nrf52 BLE 

# periodically call another function without drift
def do_every(period,f,*args):
    def g_tick():
        t = time.time()
        while True:
            t += period
            yield max(t - time.time(),0)
    g = g_tick()
    while True:
        time.sleep(next(g))
        f(*args)

# intitialise the ble communication
def ble_init():
    global flexSensorCharValue

    print("Connecting...")

    FlexSensorSuit = btle.Peripheral("29:F0:E3:F9:C9:CD")

    FlexSensorSuit.setMTU(23)

    print("connected")
    FlexSensorSuit.getServices()
    flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
    print("connected to service")

    flexSensorCharValue = flexSensorService.getCharacteristics()[0]
    print("connected to characteristic")

prevTime = 0

# fucntion to read data from ble characteristic
def ble_read(ble_queue):
    global prevTime
    print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}')
    prevTime = time.perf_counter()
    val = flexSensorCharValue.read()
    ble_queue.put(val)

def writer_task(ble_queue, f):
    prevTime_writer = 0.0
    
    writer = csv.writer(f)
    writer.writerow(["ElbowFlex", "ShoulderFlex1", "ShoulderFlex2", "ShoulderFlex3", "ForearmFlex", "HandFlex1", "HandFlex2"])
    while True:
        print(f'writer loop: {(time.perf_counter()-prevTime_writer)*1000:.3f}')
        prevTime_writer = time.perf_counter()
        # wait for and read the queue from BLE_read
        ble_val = ble_queue.get()
        #ble_val = struct.unpack("<hhhhhhh",ble_val)
        #queue_val = np.divide(ble_val,100)
        print(ble_val)
        #writer.writerow(np.divide(ble_val,100))
        #print(f'duration: {(time.perf_counter()-prevTime_writer)*1000:.3f}')


if __name__ == "__main__":

    ble_init()
    
    ble_q = Queue(maxsize=1)

    with open('Pose_Estimation/output.csv', 'w', newline='') as f:
        ble_thread = threading.Thread(target=do_every, args=(0.02, ble_read, ble_q))
        writer_thread = threading.Thread(target=writer_task, args=(ble_q, f))

        ble_thread.start()
        writer_thread.start()

        ble_thread.join()
        writer_thread.join()
