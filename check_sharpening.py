import cv2
cv2.namedWindow("Parameters")
def empty(a):
    pass
cv2.createTrackbar("ksize","Parameters",1, 50, empty)
cv2.setTrackbarPos("ksize","Parameters",5)
cv2.createTrackbar("sharpLevel","Parameters",1, 50, empty)
cv2.setTrackbarPos("sharpLevel","Parameters",1)
# cv2.createTrackbar("sigmax","Parameters",1, 50, empty)
# cv2.setTrackbarPos("sigmax","Parameters",5)
while True:
    img = cv2.imread('./inputs/cropped_license_plate.jpg')
    img = cv2.resize(img,dsize=(500,250), interpolation= cv2.INTER_CUBIC )
    ksize= cv2.getTrackbarPos("ksize","Parameters")
    # sigmax= cv2.getTrackbarPos("sigmax","Parameters")
    sharpLevel = cv2.getTrackbarPos("sharpLevel","Parameters")
    ksize= ksize*2+1
    output_blur = cv2.GaussianBlur(img, (ksize, ksize), ksize)
    output_blur1 = cv2.GaussianBlur(img, (7, 7), 7)
    output_blur2 = cv2.GaussianBlur(img, (9, 9), 9)
    output_blur3 = cv2.GaussianBlur(img, (11, 11), 11)

    mask = cv2.subtract(img, output_blur)
    mask1 = cv2.subtract(img, output_blur1)
    mask2 = cv2.subtract(img, output_blur2)
    mask3 = cv2.subtract(img, output_blur3)

    mask_inv= ~mask
    mask_inv3= ~mask3
    sharpened = cv2.addWeighted(img,(sharpLevel + 0.5), mask, -(sharpLevel-0.5), 0)
    sharpened1 = cv2.addWeighted(img, 1.5, mask3, -0.5, 0)
    # sharpened2 = cv2.addWeighted(img, 1.5, mask2, -0.5, 0)

    ret, output_thresh = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    img_edges = cv2.Canny(output_blur, 100, 100, apertureSize=3)
    cv2.imshow("Original", img)
    # cv2.imshow("Sharp1 ", sharpened1)
    cv2.imshow("Sharp ", sharpened)
    # cv2.imshow("Sharp2 ", sharpened2)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("Mask inverted", mask_inv)
    # cv2.imshow("Mask inverted3", mask_inv3)
    cv2.imshow("Blur ", output_blur)
    cv2.imshow("Threshold", output_thresh)
    cv2.imshow("Edges",img_edges)
    # cv2.imshow("Blur1 ", output_blur1)
    # cv2.imshow("Blur2 ", output_blur2)
    # cv2.imshow("Blur 3 ", output_blur3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;