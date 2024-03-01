import math
from ultralytics import YOLO
import cv2
from sort.sort import *
from util import get_car, read_license_plate, write_csv, recognize_plate, convert_dict_to_csv_file
import cvzone


def order_points(pts):
    # Step 1: Find centre of object
    center = np.mean(pts)

    # Step 2: Move coordinate system to centre of object
    shifted = pts - center

    # Step #3: Find angles subtended from centroid to each corner point
    theta = np.arctan2(shifted[:, 0], shifted[:, 1])

    # Step #4: Return vertices ordered by theta
    ind = np.argsort(theta)

    return pts[ind]
def getContours(img, orig):  # Change - pass the original image too
    biggest = np.array([])
    maxArea = 0
    imgContour = orig.copy()  # Make a copy of the original image to return
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = None
    for i, cnt in enumerate(contours):  # Change - also provide index
        area = cv2.contourArea(cnt)
        # print(f"Contour Number {i}'s Area =>{area}")
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    # Now generating the warped Image
    warped = None  # Stores the warped license plate image
    if index is not None:
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3)  # Draw the biggest contour on the image

        src = np.squeeze(biggest).astype(np.float32)  # Source points

        height = orig.shape[0]
        width = orig.shape[1]
        # Destination points
        dst = np.float32([[0, 0], [0, height - 1], [width - 1, 0], [width - 1, height - 1]])

        # Order the points correctly
        src = order_points(src)
        dst = order_points(dst)

        # Get the perspective transform
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image
        img_shape = (width, height)
        warped = cv2.warpPerspective(orig, M, img_shape, flags=cv2.INTER_LINEAR)
    #
    return biggest, imgContour, warped


results = {}

motion_tracker = Sort()

# load models
obj_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/number_plate_model.pt')

# load video
cap = cv2.VideoCapture('./inputs/masked_rotated_sample.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
vehicles = [2, 3, 5, 7]

# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("output.mp4", fourcc, 30.0, (640, 480))
#
# def empty(a):
#     pass

# read frames
frame_nmr = -1
ret = True

# Creating frame for toolbar
# def empty(a):
#     pass
# cv2.namedWindow("Param")
# cv2.resizeWindow("Param",640, 240)


# # cv2.createTrackbar("canny1", "Param" ,1, 400,empty)
# # cv2.createTrackbar("canny2", "Param",1,400,empty)
# # # cv2.createTrackbar("rect","Param",1, 20,empty)
# cv2.createTrackbar("itr","Param",1, 20,empty)
# cv2.createTrackbar("thresh","Param",1, 500,empty)
# cv2.setTrackbarPos("thresh","Param",97)


while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    # if ret and frame_nmr>=1050 and frame_nmr <=1300: 
    if ret:
        frame = cv2.resize(frame, (600, 700), interpolation=cv2.INTER_CUBIC)
        frame_clone = frame.copy()

        text = f"F:{frame_nmr}"
        print("Frame NO:", frame_nmr)

        cv2.putText(frame_clone, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)  # writing Frame number on clone frame
        results[frame_nmr] = {}

        # detect vehicles

        detections = obj_model(frame)[0]
        vehicle_detections = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                vehicle_detections.append([x1, y1, x2, y2, score])

        # Track Each Vehicle
        if len(vehicle_detections) > 1:
            vehicle_tracking_ids = motion_tracker.update(np.asarray(vehicle_detections))

            # Detecting license Plates
            license_plate_mask = cv2.imread("./inputs/license_plate_mask1.png")
            masked_image = cv2.bitwise_and(frame, license_plate_mask)
            license_plate_detections = license_plate_detector(masked_image)[0]
            for license_plate in license_plate_detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), math.ceil(score * 100) / 100
                if ((x2 - x1) * (y2 - y1) >= 1400 and (x2 - x1) * (y2 - y1) <= 2400):
                    # Assigning vehicle to license plate
                    xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id = get_car(license_plate,
                                                                                     vehicle_tracking_ids)
                    xvehicle1, yvehicle1, xvehicle2, yvehicle2 = int(xvehicle1), int(yvehicle1), int(xvehicle2), int(
                        yvehicle2)
                    if (xvehicle1, yvehicle1, xvehicle2, yvehicle2, vehicle_id) != (-1, -1, -1, -1, -1):

                        # Drawing Rectangle Around vehicle
                        cvzone.cornerRect(frame_clone,
                                          (xvehicle1, yvehicle1, (xvehicle2 - xvehicle1), (yvehicle2 - yvehicle1)))
                        cvzone.putTextRect(frame_clone, 'Vehicle', (max(xvehicle1, 0), yvehicle1),
                                           scale=0.8, thickness=1)
                        # Drawing Rectangle around license Plate
                        cvzone.cornerRect(frame_clone, bbox=(x1, y1, x2 - x1, y2 - y1), l=10, t=2, rt=1)
                        cvzone.putTextRect(frame_clone, f'License Plate', (max(x1, 0), (max(30, y1 - 20))),
                                           scale=0.8, thickness=1)

                        cropped_license_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                        # #resizing License Plate
                        cropped_license_plate = cv2.resize(cropped_license_plate, None, fx=4, fy=4,
                                                           interpolation=cv2.INTER_CUBIC)
                        # cropped_license_plate = cv2.resize(cropped_license_plate, (600,400), interpolation=cv2.INTER_CUBIC)
                        # cv2.imshow("License Plate ", cropped_license_plate)

                        gray = cv2.cvtColor(cropped_license_plate, cv2.COLOR_RGB2GRAY)
                        # cv2.imshow("Gray", gray)
                        #
                        # perform gaussian blur to smoothen image
                        blur = cv2.GaussianBlur(gray, (5, 5), 0)
                        # cv2.imshow("blur", blur)

                        imgCanny = cv2.Canny(blur, 80, 120)

                        # imgCanny= cv2.Canny(blur, canny1,canny2)
                        # cv2.imshow("Canny",imgCanny)
                        #
                        #
                        # create rectangular kernel for dilation

                        rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

                        # rect= cv2.getTrackbarPos("rect","Param")
                        # rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (rect,rect))
                        #
                        # apply dilation to make regions more clear

                        dilation = cv2.dilate(imgCanny, rect_kern, iterations=8)

                        # itr= cv2.getTrackbarPos("itr","Param")
                        # dilation = cv2.dilate(imgCanny,rect_kern,iterations=itr)
                        # cv2.imshow("Dilation", dilation)
                        #
                        # applying erosion on image dilation
                        thresh = cv2.erode(dilation, rect_kern, iterations=2)
                        # threshitr= cv2.getTrackbarPos("threshitr", "Param")
                        # # thresh = cv2.erode(dilation,rect_kern,iterations=threshitr)
                        # # ret, thresh = cv2.threshold(dilation, thresh1, thresh2, cv2.THRESH_OTSU )
                        # # ret, thresh = cv2.threshold(dilation, 0, 255, cv2.THRESH_OTSU )
                        # cv2.imshow("thresh", thresh)

                        # Getting countour and warped image on license plate
                        biggest, imgContour, warped = getContours(thresh, cropped_license_plate)
                        # cv2.imshow("Contour on license Plate is ", imgContour)
                        # cv2.namedWindow(f"Contour{frame_nmr}")
                        # cv2.resizeWindow(f"Contour{frame_nmr}",640, 240)
                        # h, w, c= imgContour.shape
                        # x= w-10
                        # y=10
                        # text= f"{frame_nmr}"
                        # cv2.circle(imgContour,(50,50),2,(0,255,0),4)
                        # cv2.putText(imgContour,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        # cv2.imshow("Contour", imgContour)

                        # cv2.imshow("Original", frame_clone)
                        # Preprocessing  warped Image
                        if type(warped) != type(None):
                            warped = cv2.resize(warped, None, fx=2, fy=1, interpolation=cv2.INTER_CUBIC)
                            # cv2.imshow("Warped", warped)
                            # Preprocessing warped Image
                            license_plate_crop_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                            # while True:
                            # thresh_val= cv2.getTrackbarPos("thresh","Param")
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 96, 255,
                                                                         cv2.THRESH_BINARY_INV)
                            # filtered_img= cv2.medianBlur(license_plate_crop_thresh,3)
                            # cv2.imshow("Filtered Image ", filtered_img)

                            # read license plate
                            # license_plate_text,license_plate_text_score= read_license_plate(license_plate_crop_thresh)
                            number_contours, hierarchy = cv2.findContours(license_plate_crop_thresh, cv2.RETR_EXTERNAL,
                                                                          cv2.CHAIN_APPROX_NONE)
                            license_plate_crop_thresh = cv2.cvtColor(license_plate_crop_thresh, cv2.COLOR_GRAY2BGR)
                            # avg_area=0
                            for i, cnt in enumerate(number_contours):
                                area = int(cv2.contourArea(cnt))
                                x, y, w, h = cv2.boundingRect(cnt)
                                x, y, w, h = int(x), int(y), int(w), int(h)
                                if w * h > 5000 or w * h < 800:
                                    cv2.drawContours(license_plate_crop_thresh, number_contours, i, (0, 0, 0),
                                                     cv2.FILLED)

                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                            if license_plate_text is not None:
                                cvzone.putTextRect(license_plate_crop_thresh, f'{license_plate_text}', (0, 30), 2,
                                                   thickness=2)
                                results[frame_nmr][vehicle_id] = {'car': {'bbox': [xvehicle1, yvehicle1, xvehicle2, yvehicle2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}
                                # if frame_nmr == 1817:
                                #     print(f'Result for frame Number[{frame_nmr}] is {results[frame_nmr]}')

        cv2.imshow('Frame',frame_clone)
        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            exit()
# for frame_nmr in results.keys():
#     print(f"Frame Number[{frame_nmr}]:")
#     for car_id in results[frame_nmr].keys():
#         if 'car' in results[frame_nmr][car_id].keys() and \
#                 'license_plate' in results[frame_nmr][car_id].keys() and \
#                 'text' in results[frame_nmr][car_id]['license_plate'].keys():
#             print("yes its here")
#         print(f'{car_id}=>{results[frame_nmr][car_id]}')

# Writing Result to CSV File
# write_csv(results, './test.csv')
# convert_dict_to_csv_file(results,'./test.csv')


# license_plate_text,license_plate_text_score= read_license_plate(license_plate_crop_thresh)
#
# print(license_plate_text,license_plate_text_score)
#
#             print(vehicle_detections)
#             cv2.imshow("Image",frame_clone)
#             cv2.waitKey(0)
#
#             # crop license plate from frame
#             license_plate_crop= frame[int(y1):int(y2), int(x1):int(x2)]
#
#
#             #process the license Plate
#             license_plate_crop_gray=cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY)
#             _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
#
#             #read license plate
#             license_plate_text,license_plate_text_score= read_license_plate(license_plate_crop_thresh)
#
#             license_plate_text, license_plate_text_score= recognize_plate(frame,(x1,y1,x2,y2))
#
#             print("\n\n###License Plate: ",license_plate_text,license_plate_text_score, "####\n\n\n")
#
#             if license_plate_text is not None:
#                 print("#$#$#Writing Result#$#$#$#")
#                 results[frame_nmr] = {'car': {'bbox': [ xvehicle1, yvehicle1, xvehicle2, yvehicle2]},
#                                                 'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                 'text': license_plate_text,
#                                                                 'bbox_score': score,
#                                                                 'text_score': license_plate_text_score}}
#                 # current_time = datetime.datetime.now()
#                 # entry_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
#                 # results[itr]={{'Number':license_plate_text},{'Time':entry_time}}
#
# write_csv(results, './test.csv')
# convert_dict_to_csv_file(results,'./test.csv')


# -----------------------------------------------------------------------------------------------------------------
#         results[frame_nmr] = {}
#         detect vehicles
#
#         detections = coco_model(frame)[0]
#         detections_ = []
#         for detection in detections.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = detection
#             if int(class_id) in vehicles:
#                 detections_.append([x1, y1, x2, y2, score])
#
#         # track vehicles
#         track_ids = mot_tracker.update(np.asarray(detections_))
#
#         # detect license plates
#         license_plates = license_plate_detector(frame)[0]
#         for license_plate in license_plates.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = license_plate
#
#             # assign license plate to car
#             xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
#
#             if car_id != -1:
#
#                 # crop license plate
#                 license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
#
#                 # process license plate
#                 license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
#                 _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
#
#                 # read license plate number
#                 license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
#
#                 if license_plate_text is not None:
#                     results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
#                                                   'license_plate': {'bbox': [x1, y1, x2, y2],
#                                                                     'text': license_plate_text,
#                                                                     'bbox_score': score,
#                                                                     'text_score': license_plate_text_score}}
# write results
# write_csv(results, './test.csv')
