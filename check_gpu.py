import math
from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv, recognize_plate, convert_dict_to_csv_file
import cvzone

motion_tracker = Sort()

# load models
obj_model = YOLO('yolov8m.pt')
names= obj_model.names

# load video
cap = cv2.VideoCapture('./inputs/masked_rotated_sample.mp4')
fps= cap.get(cv2.CAP_PROP_FPS)
# 2: car, 3: motorcycle
vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    # if ret and frame_nmr>=1050 and frame_nmr <=1300:
    if ret and frame_nmr > 1000:
        frame = cv2.resize(frame, (600, 700), interpolation=cv2.INTER_CUBIC)
        frame_clone = frame.copy()

        text = f"F:{frame_nmr}"
        print("Frame NO:", frame_nmr)

        cv2.putText(frame_clone, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)  # writing Frame number on clone frame

        # results[frame_nmr] = {}

        # detect vehicles
        detections = obj_model(frame)[0]
        vehicle_detections = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                area = int(x2-x1) * int(y2-y1)
                if area > 1000:
                    vehicle_detections.append([x1, y1, x2, y2, score])
                    x1,x2,y1,y2= int(x1),int(x2),int(y1),int(y2)
                    cvzone.cornerRect(frame_clone, (x1, y1, (x2 - x1), (y2 - y1)))
                    cvzone.putTextRect(frame_clone, f'{class_id}:{names[class_id]}', (x1, y1 - 20), scale=1, thickness=1)

        # Track Each Vehicle
        # if len(vehicle_detections) > 1:
        #     vehicle_tracking_ids = motion_tracker.update(np.asarray(vehicle_detections))
        #
        #     for vehicle in vehicle_tracking_ids:
        #         x1, y1, x2, y2, id = vehicle
        #         x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
        #         cvzone.cornerRect(frame_clone, (x1, y1, (x2 - x1), (y2 - y1)))
        #         cvzone.putTextRect(frame_clone, f'{id}', (x1,y1-20),scale=1, thickness=1)

        cv2.imshow("Frame", frame_clone)
        k= cv2.waitKey(1)
        if k & 0xFF == ord('q'):
            exit()
        if k == 32:
            cv2.waitKey(0)



