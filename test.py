from ultralytics import YOLO
import cv2
import math
from sort.sort import *
import cvzone
motion_tracker = Sort()
obj_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('./models/number_plate_model_2.pt')


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
            approx = cv2.approxPolyDP(cnt,0.05*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
                index = i  # Also save index to contour

    # Now generating the warped Image
    warped = None  # Stores the warped license plate image
    if index is not None:
        cv2.drawContours(imgContour, contours, index, (255, 0, 0), 3) # Draw the biggest contour on the image

        src = np.squeeze(biggest).astype(np.float32) # Source points

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

    return biggest, imgContour, warped


#Reading the video capture
cap = cv2.VideoCapture('./inputs/sample.mp4')

# Get the dimensions of the input video
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps= cap.get(cv2.CAP_PROP_FPS)

video = cv2.VideoWriter("./outputs/output.mp4",cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS), (600,700))

vehicles = [2, 3, 5, 7]
ret = True
frame_nmr=0


while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    # if ret and frame_nmr>=1050 and frame_nmr <=1300:
    if ret and frame_nmr> 385:
        frame = cv2.resize(frame, (600,700), interpolation=cv2.INTER_CUBIC)
        mask_vehicle= cv2.imread('inputs/mask_vehicle.png')
        frame = cv2.bitwise_and(frame,mask_vehicle)
        frame_clone = frame.copy()

        text= f"F:{frame_nmr}"
        print("Frame NO:", frame_nmr)

        cv2.putText(frame_clone,text,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2) # writing Frame number on clone frame

        # Detecting Vehicles
        mask_plate = cv2.imread('inputs/mask_licenseplate2.png')
        frame = cv2.bitwise_and(frame, mask_plate)
        detections = obj_model(frame)[0]
        vehicle_detections = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            x1, y1, x2, y2, score = int(x1), int(y1), int(x2), int(y2), math.ceil(score*100)/100
            if int(class_id) in vehicles:
                area = (y2-y1)*(x2-x1)

                vehicle_detections.append([x1, y1, x2, y2, score])
                # cv2.rectangle(frame_clone, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 3)  # Drawing Rectangle Around vehicle
                cvzone.cornerRect(frame_clone, bbox=(x1, y1, (x2-x1), (y2-y1)),l= 20, t=2, rt=2)
                cvzone.putTextRect(frame_clone,f'Area:{area}', (max(x1,0),max(30, y1-20)), scale= 0.8, thickness=1)

        # Detecting license Plates
        license_plate_detections = license_plate_detector(frame)[0]
        for license_plate in license_plate_detections.boxes.data.tolist():
            x1,y1,x2,y2, score, class_id= license_plate
            x1, y1, x2, y2 , score = int(x1), int(y1), int(x2),int(y2), math.ceil(score*100)/100
            area = (x2-x1)*(y2-y1)
            if area > 3000:
                continue
            cvzone.cornerRect(frame_clone, bbox=(x1, y1, x2-x1, y2-y1), l=10, t=2, rt=1)
            num_plate = frame[y1:y2,x1:x2]
            num_plate= cv2.resize(num_plate, (500,250) , interpolation=cv2.INTER_CUBIC)
            gray_num_plate= cv2.cvtColor(num_plate,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_num_plate,(5,5),0)
            # cv2.imshow("Blur Image", blur)
            mask = cv2.subtract(gray_num_plate, blur)
            cv2.imshow("Plate Image ", num_plate)
            sharpened_image = cv2.addWeighted(gray_num_plate,1.5, mask,-5,0 )
            ret, imgf = cv2.threshold(sharpened_image ,127, 255, cv2.THRESH_BINARY_INV, cv2.THRESH_OTSU)

            h,w = imgf.shape[:2]
            centerX, centerY= (w//2,h//2)
            # get rotation matrix
            M = cv2.getRotationMatrix2D((centerX, centerY), -20, 1.0)
            rotated = cv2.warpAffine(imgf, M ,(w,h))

            # cropping the image for better view
            cropped_license_plate = rotated[50:h-50,20:w]
            #finding contour in cropped license plate
            contours, hierarchy = cv2.findContours(cropped_license_plate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cropped_license_plate= cv2.cvtColor(cropped_license_plate,cv2.COLOR_GRAY2BGR)
            cv2.drawContours(cropped_license_plate, contours, -1, (0, 0, 255), 2)
            h,w = cropped_license_plate.shape[:2]
            centerX, centerY = (w // 2, h // 2)
            cv2.line(cropped_license_plate,(0,centerY) ,(w,centerY),(0,255,0),2 )
            for contour in contours:
                # if cv2.contourArea(contour) > 1500:
                #     cv2.drawContours(cropped_license_plate,[contour],-1,(0,0,0),-1)
                # Get the bounding rectangle of the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Draw the bounding box on the image
                # cv2.rectangle(cropped_license_plate, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cvzone.putTextRect(cropped_license_plate,f'Area:{cv2.contourArea(contour)}',(x,max(20,y-10)),1,2)
                cv2.imshow("Contoured Image", cropped_license_plate)
                # cv2.waitKey(0)
        cv2.imshow("Frame", frame_clone)
        k= cv2.waitKey(int(1000/fps))
        if(k & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            exit()
        if k== 32:
            cv2.waitKey(0)
        # video.write(frame_clone)