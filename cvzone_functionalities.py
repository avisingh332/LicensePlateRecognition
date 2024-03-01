# import cv2
# import cvzone
# import math
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
#
# cap= cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
#
# ret= True
#
# while ret:
#     ret, img = cap.read()
#     if ret:
#         detections= model(img)[0]
#         for detection in detections.boxes.data.tolist():
#             x1,y1,x2,y2, score, class_id= detection
#             x1, y1, x2, y2= int(x1), int(y1), int(x2), int(y2)
#             # cv2.rectangle(img,(x1,y1), (x2,y2),(255,0,255),3)
#             cvzone.cornerRect(img=img,bbox=(x1,y1, x2-x1, y2-y1) ,l= 10, t=2, rt=1)
#             score = math.ceil((score*100))/100
#             cvzone.putTextRect(img,f'{score}',(max(x1,0),(max(30,y1-20))))
#
#         # for box in detections.boxes:
#         #     x1,y1,x2,y2, score, class_id= box.xyxy[0]
#         #     x1, y1, x2, y2= int(x1), int(y1), int(x2), int(y2)
#         #     # cv2.rectangle(img,(x1,y1), (x2,y2),(255,0,255),3)
#         #     cvzone.cornerRect(img,bbox=(x1,y1, x2-x1, y2-y1))
#         #     conf = box.conf[0]
#         #     print(conf)
#         cv2.imshow("Webcam", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             exit()


import cv2

num_of_devices = cv2.cuda.getCudaEnabledDeviceCount()
print(num_of_devices)