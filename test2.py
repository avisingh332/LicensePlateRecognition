import cv2
import cvzone
import numpy as np
import matplotlib.pyplot as plt
import math
from util import read_license_plate
img = cv2.imread('./inputs/cropped_license_plate.jpg')
img = cv2.resize(img, dsize=(500,250),  interpolation=cv2.INTER_CUBIC)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
output_blur = cv2.GaussianBlur(img,(5,5),5)
mask = cv2.subtract(img, output_blur)
sharpened = cv2.addWeighted(img,1.5, mask,-0.5,0)
ret, output_thresh= cv2.threshold(sharpened,127, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

# img_edges= cv2.Canny(output_thresh, 100, 100, apertureSize=3)
img_edges= cv2.Canny(output_thresh, 100, 100, apertureSize=3)

lines = cv2.HoughLines(img_edges,1, math.pi/180 ,100)
linesP = cv2.HoughLinesP(img_edges, 1,theta= math.pi / 180.0,threshold= 100, minLineLength=10, maxLineGap=10)

img_lines= cv2.cvtColor(img_edges,cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        # print(f"Line is:{line}")
        rho ,theta= line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(output_thresh, (x1, y1), (x2, y2), (255, 255, 255), 15)

# cv2.imshow("output Thresh with line ", output_thresh)

angles =[]
if linesP is not None:
    for line in linesP:
        for x1, y1, x2, y2 in line:
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

# determining skew Angle
print(f"Set: {set(angles)}, Count : {angles.count}")
if len(linesP)>0:
    skew_angle= max(set(angles), key = angles.count)
else:
    skew_angle= 0

h, w = output_thresh.shape[:2]
centerX, centerY = (w//2, h//2)
# get rotation matrix
M = cv2.getRotationMatrix2D((centerX, centerY), skew_angle, 1.0)
rotated = cv2.warpAffine(output_thresh, M, (w, h))


horizontal_projection = np.sum(rotated, axis=1)
threshold_value= 70/100 * max(horizontal_projection);
roi= np.where(horizontal_projection >= int(threshold_value))[0]



final_cropping = rotated[roi.min():roi.max(),:]


# cv2.imshow("Edges", img_edges)
# cv2.imshow("lines",img_lines)
# cv2.imshow("Rotated", rotated)


# Vertical and Horizontal Projection Analysis
# horizontal_projection = np.sum(final_cropping,axis=1)
# vertical_projection = np.sum(final_cropping, axis=0)
#
# height, width = final_cropping.shape[:2]
#
# blankImage1 = np.zeros((height, width, 3), np.uint8)
# blankImage2 = np.zeros((height, width, 3), np.uint8)
#
# for row in range(height):
#     cv2.line(blankImage1, (width, row), (width-int(horizontal_projection[row]/255),row), (255,255,255), 1)
#
# for col in range(width):
#     cv2.line(blankImage2, (col, 0), (col, int(vertical_projection[col]/255)), (255,255,255), 1)

kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
img_dilation = cv2.dilate(final_cropping, kernal, iterations=2)
contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img_contours = cv2.cvtColor(img_dilation, cv2.COLOR_GRAY2BGR)

longest_contour = None
longest_perimeter = 0
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    if perimeter > longest_perimeter:
        longest_contour = contour
        longest_perimeter = perimeter

# Draw the longest contour on the image
cv2.drawContours(img_contours, [longest_contour], -1, (255, 255, 255), 4)
mask= np.zeros_like(img_contours)
cv2.drawContours(mask, [longest_contour],-1, (255,255,255), -1)
mask_inv = 255 -mask

img_contours = cv2.add(img_contours,mask_inv)

# img_contours = ~img_contours
img_contours = cv2.erode(img_contours,kernal,iterations=2)
img_ocr = cv2.GaussianBlur(img_contours,(3,3),3)
cv2.imshow("ocr ", img_ocr)
license_plate_text, license_plate_text_score = read_license_plate(img_ocr)
print(f"Text:{license_plate_text}, Score: {license_plate_text_score}")
# print( cv2.arcLength(max_len_contour, True))
# longest_contour= contours[0]
# if contours is not None:
#     for cnt in contours:
#         arc_len = cv2.arcLength(cnt, True)
#         if arc_len > max_len_contour:
#             max_len_contour = arc_len
#             longest_contour = cnt
        # approx = cv2.approxPolyDP(cnt, epsilon=1.0* cv2.arcLength(cnt, False), closed=False)
        # if len(approx)==4:
        #     x,y,w,h = cv2.boundingRect(cnt)
        #     cv2.rectangle(img_contours,(x,y), (x+w,y+h), (0,0,255), 2)

# cv2.drawContours(img_contours,[longest_contour], -1, (0, 0,  255),3)

# x, y, w, h = cv2.boundingRect(longest_contour)
# cv2.rectangle(img_contours, (x, y), (x+w, y+h), (0, 0, 255), 2)
# print(f"Approx=>{approx}")


# cv2.imshow("Dilated Image", img_dilation)
cv2.imshow("Contours", img_contours)
# cv2.imshow("Deskewed Image", rotated)
# cv2.imshow("Final Cropping", final_cropping)
cv2.waitKey(0)
# plt.subplot(221)
# plt.imshow(final_cropping,cmap='gray')
# plt.subplot(222)
# plt.imshow(blankImage1,cmap='gray')
# plt.subplot(223)
# plt.imshow(blankImage2, cmap='gray')
#
# plt.show()
#
# k = cv2.waitKey(0)





# plt.subplot(2,2,1)
# plt.imshow(img_edges, cmap='gray')
#
# plt.subplot(2,2,2)
# plt.imshow(img_lines)
# plt.show()

# rotateImage(img)
#
# plt.subplot(3,2,1)
# plt.title("Original")
# plt.imshow(img,cmap='gray')
#
# plt.subplot(3,2,2)
# plt.title("Blur")
# plt.imshow(output_blur,cmap='gray')
#
# plt.subplot(3,2,3)
# plt.title("Sharp")
# plt.imshow(sharpened,cmap='gray')
#
# plt.subplot(3,2,4)
# plt.title("Threholded")
# plt.imshow(output_thresh,cmap='gray')
#
# plt.show()
# #
# #
# # #sharpening using add weighted()
# # sharpened1= cv2.addWeighted(img, 1.5, output_blur, -0.5,0)
# # # sharpened2= cv2.addWeighted(img, 3.5, output_blur, -2.5,0 )
# # sharpened3 = cv2.addWeighted(img, 7.5, output_blur, -6.5,0 )
#
# # sharpened5 = cv2.addWeighted(img ,2.5, mask,-1.5,0)
# #
# # # sharpened5 = cv2.addWeighted(img ,4.5, output_blur ,-3.5 ,0)
# # # sharpened6 = cv2.addWeighted(img ,7.5, mask ,-6.5 ,0)
# # #removing noise using medianBlur
# # output_med= cv2.GaussianBlur(sharpened3,(5,5), 2)
#
# # ret, output_sharp2= cv2.threshold(sharpened4 ,127, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
# #
# #
# # # cv2.imshow("Sharpened1", sharpened1)
# # # cv2.imshow("Sharpened2", sharpened2)
# # # cv2.imshow("Sharpened3", sharpened3)
# # # cv2.imshow("Original Image ", img)
# # # cv2.imshow("Blur Image ", output_blur)
# # # cv2.imshow("Median Blur", output_med)
# # cv2.imshow("Output Threshold", output_sharp)
# # cv2.imshow("Output Threshold 1", output_sharp2)
# #
# # # cv2.imshow("Mask", mask)
# # cv2.imshow("Sharpened 4", sharpened4)
# # cv2.imshow("Sharpened 5", sharpened5)
# #
# # # cv2.imshow("Sharpened 5", sharpened5)
# # # cv2.imshow("Sharpened 6", sharpened6)
# #
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()