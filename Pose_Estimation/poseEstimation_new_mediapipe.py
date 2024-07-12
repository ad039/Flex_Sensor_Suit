import mediapipe as mp
import cv2
import time

model_path = 'Pose_Landmarker/pose_landmarker_lite.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    #print('pose landmarker result: {}'.format(result))
    return

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with PoseLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
  # ...
    cap = cv2.VideoCapture(0)
    time_prev = 0

    while cap.isOpened():
        success, frame = cap.read()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        landmarker.detect_async(mp_image, int(time.time() * 1000))

        time_now = time.time()
        totalTime = time_now-time_prev
        time_prev = time_now
        fps = 1 / totalTime

        cv2.putText(frame, f'{int(fps)}', (20,70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0))

        cv2.imshow('Window', frame)

        if cv2.waitKey(5) & 0xFF ==27:
                break