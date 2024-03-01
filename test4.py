import numpy as np
from ultralytics import YOLO
import cv2
import math
from sort.sort import *
import cvzone
from util import read_license_plate, write_cvs1

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

results = {}
#Reading the video capture
cap = cv2.VideoCapture('./inputs/sample.mp4')

# Get the dimensions of the input video
frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps= cap.get(cv2.CAP_PROP_FPS)
# print(f"{frame_width}x{frame_height}")
# exit()
video = cv2.VideoWriter("./outputs/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (600, 700))
# frames_with_plate= [662]
frames_with_plate = [397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 496, 497, 498, 499, 500, 501, 502, 503, 504, 658, 658, 659, 659, 660, 661, 662, 663, 666, 777, 780, 781, 782, 783, 784, 785, 786, 841, 842, 843, 844, 845, 846, 913, 965, 966, 967, 968, 969, 970, 1098, 1099, 1100, 1101, 1102, 1238, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246]

vehicles = [2, 3, 5, 7]
ret = True
frame_nmr = 0
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    # if ret and frame_nmr>=1050 and frame_nmr <=1300:
    if ret and frame_nmr in frames_with_plate and frame_nmr > 600:
        # frame = cv2.resize(frame, (600,700), interpolation=cv2.INTER_CUBIC)
        mask_plate = cv2.imread('inputs/mask_licenseplate_960x1072.jpg')
        frame = cv2.bitwise_and(frame, mask_plate)
        frame_clone = frame.copy()

        text= f"F:{frame_nmr}"
        print("Frame NO:", frame_nmr)
        cv2.putText(frame_clone, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # writing Frame number on clone frame

        # Detecting license Plates

        license_plate_detections = license_plate_detector(frame)[0]
        for license_plate in license_plate_detections.boxes.data.tolist():
            x1,y1,x2,y2, score, class_id= license_plate
            x1, y1, x2, y2 , score = int(x1), int(y1), int(x2),int(y2), math.ceil(score*100)/100
            area = (x2-x1)*(y2-y1)
            cvzone.putTextRect(frame_clone,f'Area:{area}',pos=(0, frame_clone.shape[0]-10),scale=2)
            cvzone.cornerRect(frame_clone, bbox=(x1, y1-5, (x2+5)-x1, y2-(y1-5)), l=10, t=2, rt=1)
            num_plate = frame[y1-5:y2,x1:x2+5]
            num_plate= cv2.resize(num_plate, (700,350) , interpolation=cv2.INTER_CUBIC)

            img = cv2.cvtColor(num_plate, cv2.COLOR_BGR2GRAY)
            output_blur = cv2.GaussianBlur(img, (5, 5), 0)
            mask = cv2.subtract(img, output_blur)
            sharpened = cv2.addWeighted(img, 1.5, mask, -0.5, 0)

            ret, output_thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

            img_edges = cv2.Canny(output_thresh, 100, 100, apertureSize=3)
            img_edges1 = cv2.Canny(sharpened, 90, 140, apertureSize=3)
            lines = cv2.HoughLines(img_edges, 1, math.pi / 180, 100)

            linesP = cv2.HoughLinesP(img_edges, 1, theta=math.pi / 180.0, threshold=100, minLineLength=10,
                                     maxLineGap=10)
            img_lines = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)

            # output_thresh=cv2.cvtColor(output_thresh, cv2.COLOR_GRAY2BGR)

            angles = []
            if linesP is not None:
                for line in linesP:
                    for x1, y1, x2, y2 in line:
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        angles.append(angle)

                # determining skew Angle
                # print(f"Set: {set(angles)}, Count : {angles.count}")
                angles = [int(element) for element in angles]
                if len(linesP) > 0:
                    skew_angle = max(set(angles), key=angles.count)
            else:
                skew_angle = 0

            # For Visualizing  Lines Detected
            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * -b)
                    y1 = int(y0 + 1000 * a)
                    x2 = int(x0 - 1000 * -b)
                    y2 = int(y0 - 1000 * a)
                    if int(theta)== skew_angle:
                        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 5)
                        cv2.line(output_thresh, (x1, y1), (x2, y2), (255, 255, 255), 15)

            print(f'Set of Angle=> {set(angles)}\n\n Skew Angle=> {skew_angle}')
            h, w = output_thresh.shape[:2]
            centerX, centerY = (w // 2, h // 2)
            # get rotation matrix
            M = cv2.getRotationMatrix2D((centerX, centerY), skew_angle, 1.0)
            rotated = cv2.warpAffine(output_thresh, M, (w, h))
            # img_erosion= cv2.erode(rotated, kernal, iterations=2 )

            # Finding Horizontal Projection and then doing final cropping
            horizontal_projection = np.sum(rotated, axis=1)
            threshold_value = 70 / 100 * max(horizontal_projection)
            roi = np.where(horizontal_projection >= int(threshold_value))[0]
            final_cropping = rotated[roi.min():roi.max(), :]


            # Fiding contour on final cropping
            kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            img_dilation = cv2.dilate(final_cropping, kernal, iterations=2)
            contours, hierarchy = cv2.findContours(final_cropping, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            img_contours = cv2.cvtColor(final_cropping, cv2.COLOR_GRAY2BGR)
            max_len_contour = cv2.arcLength(contours[0], closed=True)
            longest_contour = contours[0]
            for i, cnt in enumerate(contours):
                cnt_len = cv2.arcLength(cnt, closed=True)
                if cnt_len > max_len_contour:
                    max_len_contour = cnt_len
                    longest_contour = cnt

            mask = np.zeros_like(img_contours)
            cv2.drawContours(mask, [longest_contour], -1, (255, 255, 255), -1)
            cv2.drawContours(img_contours, [longest_contour], -1, (255, 255, 255), 3)
            mask_inv = 255 - mask
            img_contours = cv2.add(img_contours, mask_inv)
            img_contours = cv2.erode(img_contours, kernal, iterations=1)
            img_contours_inv = ~cv2.cvtColor(img_contours, cv2.COLOR_BGR2GRAY)
            contours, hierarchy = cv2.findContours(img_contours_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.imwrite(f"./outputs/{frame_nmr}.jpg", frame_clone)
            for i, cnt in enumerate(contours):
                if cv2.contourArea(cnt) < 800:
                    cv2.drawContours(img_contours, contours, i, (255, 255, 255), -1)

            # img_ocr = cv2.GaussianBlur(img_contours, (3, 3), 3)
            # license_plate_text, license_plate_text_score = read_license_plate(img_ocr)
            # if license_plate_text is not None:
            #     results[frame_nmr] = {'license_plate': {'bbox': [x1, y1, x2, y2],
            #                                             'text': license_plate_text,
            #                                             'bbox_score': score,
            #                                             'text_score': license_plate_text_score}}
            #     print(f'{license_plate_text} Score:{int(license_plate_text_score)}')
            #     cvzone.putTextRect(img_contours, f'{license_plate_text} Length:{len(license_plate_text)}', (0, 20),
            #                        scale=2)
            # frames_with_plate.append(frame_nmr)
            cv2.imshow("Final Cropping", final_cropping)
            # cv2.imshow("Rotated", rotated)
            # cv2.imshow("num_Plate", num_plate)
            cv2.imshow("contours", img_contours)
            # cv2.imshow("Sharpened", sharpened)
            # cv2.imshow("Threshold", output_thresh)

            cv2.imshow("Lines ", img_lines)
            # cv2.imshow("Images with Edges", img_edges)

        frame_clone = cv2.resize(frame_clone,dsize=(600,700), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Original", frame_clone)

        # key = cv2.waitKey(int(1000 / fps))
        key = cv2.waitKey(0)
        if key == 32:
            cv2.waitKey(0)
        elif key & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
