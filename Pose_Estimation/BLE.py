from bluepy import btle
import numpy as np
import struct
import time
import threading
from queue import Queue
import csv

#"29:F0:E3:F9:C9:CD" for Arduino Nano BLE
#"93:43:92:07:91:11" for Xioa nrf52 BLE 

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


print("Connecting...")

FlexSensorSuit = btle.Peripheral("29:F0:E3:F9:C9:CD")

print("connected")
FlexSensorSuit.getServices()
flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
print("connected to service")

flexSensorCharValue = flexSensorService.getCharacteristics()[0]
print("connected to characteristic")
i = 0
prevTime = 0


def ble_read(ble_queue):
    prevTime = time.perf_counter()
    val = flexSensorCharValue.read()
    val = struct.unpack("<hhhhhhh",val)
    ble_queue.put(val, 100)

def writer_task(ble_queue):
    prevTime = 0.0
    with open('Pose_Estimation/output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ElbowFlex", "ShoulderFlex1", "ShoulderFlex2", "ShoulderFlex3", "ForearmFlex", "HandFlex1", "HandFlex2", "P_x", "P_y", "P_z", "O_x", "O_y", "O_z"])
        while True:
            print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}')
            prevTime = time.perf_counter()
            # read the queues from BLE and Pose Estimation
            ble_val = ble_queue.get()
            queue_val = np.divide(ble_val,100)
            print(queue_val)
            writer.writerow(queue_val)
            #print(f'duration: {(time.perf_counter()-prevTime)*1000:.3f}')


if __name__ == "__main__":
    
    ble_q = Queue(maxsize=100)

    ble_thread = threading.Thread(target=do_every, args=(0.05, ble_read, ble_q))
    writer_thread = threading.Thread(target=writer_task, args=(ble_q,))

    ble_thread.start()
    writer_thread.start()

    ble_thread.join()
    writer_thread.join()
