from bluepy import btle
import numpy as np
import struct
import time

#"29:F0:E3:F9:C9:CD" for Arduino Nano BLE
#"93:43:92:07:91:11" for Xioa nrf52 BLE 

print("Connecting...")

FlexSensorSuit = btle.Peripheral("93:43:92:07:91:11")

print("connected")
FlexSensorSuit.getServices()
flexSensorService = FlexSensorSuit.getServiceByUUID("0000fff0-0000-1000-8000-00805f9b34fb")
print("connected to service")

flexSensorCharValue = flexSensorService.getCharacteristics()[0]
print("connected to characteristic")
i = 0
prevTime = 0.0

while True:
    if time.perf_counter() - prevTime > 0.09:
        print(f'loop: {(time.perf_counter()-prevTime)*1000:.3f}')
        prevTime = time.perf_counter()
        val = flexSensorCharValue.read()
        val = struct.unpack("<hhhhhhh",val)
        #print(val)
        print(f'duration: {(time.perf_counter()-prevTime)*1000:.3f}')