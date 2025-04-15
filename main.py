import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from util import get_parking_spots_bboxes, empty_or_not

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Load resources
mask_path = r'C:\Users\Rakesh\OneDrive\Desktop\Codes\Python\Yolov8 models\03__Security\Media\mask.jpg'
video_path = r'C:\Users\Rakesh\OneDrive\Desktop\Codes\Python\Yolov8 models\03__Security\Media\vid.mp4'

mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)

connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

spots_status = [None for _ in spots]
diffs = [None for _ in spots]
previous_frame = None
frame_nmr = 0
step = 30

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    yolo_results = model.track(frame, persist=True)
    frame_with_objects = yolo_results[0].plot()  # Draw object boxes

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        print([diffs[j] for j in np.argsort(diffs)][::-1])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_index in arr_:
            spot = spots[spot_index]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_index] = spot_status

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_index, spot in enumerate(spots):
        spot_status = spots_status[spot_index]
        x1, y1, w, h = spot
        overlay = frame_with_objects.copy()
        color = (0, 255, 0) if not spot_status else (0, 0, 255)
        cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), color, -1)
        alpha = 0.4
        frame_with_objects = cv2.addWeighted(overlay, alpha, frame_with_objects, 1 - alpha, 0)

    # Show number of available spots (optional)
    # cv2.rectangle(frame_with_objects, (80, 20), (550, 80), (0, 0, 0), -1)
    # cv2.putText(frame_with_objects, f'Available spots: {sum(spots_status)} / {len(spots_status)}',
    #             (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 800, 900)
    cv2.imshow('frame', frame_with_objects)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
