# Flex_Sensor_Suit
Flex Sensor Suit Repository based on Undergraduate Thesis by Alex Dunn

This repository is set up to use venv with python3, you can install the required python3 libraries to your virtual environment using
```
pip install -r "*_requirements.txt"
```
There are different virtual environments for both Pose Estiamtion and PyTorch sections

The repository is broken into three sections:
- [PlatformIO](https://github.com/ad039/Flex_Sensor_Suit#platformio)
- [Pose_Estimation](https://github.com/ad039/Flex_Sensor_Suit#pose_estimation)
- [PyTorch](https://github.com/ad039/Flex_Sensor_Suit#pytorch)

### PlatformIO
The PlatformIO folder houses the platformIO c++ project FlexSensorSuit BLE for an mbed nrf52 microconrtoller such as the Arduino nano 33 BLE or the Seeed Xiao BLE. To run the program, platformIO will need to be installed on Visual Studio Code. Alternativley, the main.cpp file code can be coppied and pasted into the Arduino IDE.

### Pose_Estimation
In the Pose Estimation folder are python scripts to generate pose landmarks using the mediapipe pose framework, python scripts to communicate with the Arduino nano 33 BLE or Seeed Xiao BLE using Bleutooth Low Energy(BLE) and a script to perform both operations for simultaneous data collection.

### PyTorch
Inside the PyTorch folder are scripts to implement different neural network types. The sensor data is trained to predict the x, y, z position of the hand, based on mocap data. Different mocap data files are available in the 'data' folder included.

### Results (Aug 2024)

Results for a Mediapipe Pose Estimation trained neural network consisting of 7 input sensors and three outputs: the (x, y, z) position of the hand 

![circle_pose_estimation_training_results_2D](https://github.com/user-attachments/assets/9951bfaa-8181-40ad-b726-59cdc785c679)
