# Flex_Sensor_Suit
Flex Sensor Suit Repository based on Undergraduate Thesis by Alex Dunn

Installation:
Download the repository using 
```
git clone https://github.com/ad039/Flex_Sensor_Suit.git
```

This repository is set up to use venv with python3, you can install the required python3 libraries to your virtual environment using
```
pip install -r "requirements.txt"
```

The repository is broken into three sections:
- [PlatformIO](https://github.com/ad039/Flex_Sensor_Suit#platformio)
- [Pose_Estimation](https://github.com/ad039/Flex_Sensor_Suit#pose_estimation)
- [PyTorch](https://github.com/ad039/Flex_Sensor_Suit#pytorch)

### PlatformIO
The PlatformIO folder houses the platformIO c++ project FlexSensorSuit BLE for an mbed nrf52 microconrtoller such as the Arduino nano 33 BLE or the Seeed Xiao BLE. To run the program, platformIO will need to be installed on Visual Studio Code

### Pose_Estimation
In the Pose Estimation folder are python scripts to generate pose landmarks using the mediapipe pose framework, python scripts to communicate with the Arduino nano 33 BLE or Seeed Xiao BLE and a script to perform both operations.

### PyTorch
Inside the PyTorch folder is a script to read a csv file and train a basic FeedForward Neural Network. The network is then tested against an unseen snippet of data and root mean square error is calculated for the predictions.


