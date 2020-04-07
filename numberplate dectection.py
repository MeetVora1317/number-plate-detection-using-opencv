def main():

    import numpy as np

    import cv2

    import imutils

    import sys

    import pytesseract

    import pandas as pd

    import time


    #Add this line to assert the path. Else TesseractNotFoundError will be raised.

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    #Read the original image.
    ''' put the image of car with number plate below '''
    img = cv2.imread("car.jpg")

    #Using imutils to resize the image.

    img = imutils.resize(img, width=500)

    #Convert from colored to Grayscale.

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Applying Bilateral Filter on the grayscale image.


    #It will remove noise while preserving the edges. So, the number plate remains distinct.

    gray_img = cv2.bilateralFilter(gray_img, 11, 17, 17)

    #Finding edges of the grayscale image.

    c_edge = cv2.Canny(gray_img, 170, 200)

    #Finding contours based on edges detected.

    cnt, new = cv2.findContours(c_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #Storing the top 30 edges based on priority

    cnt = sorted(cnt, key = cv2.contourArea, reverse = True)[:30]

    NumberPlateCount = None

    im2 = img.copy()

    cv2.drawContours(im2, cnt, -1, (0,255,0), 3)

    count = 0

    for c in cnt:

        perimeter = cv2.arcLength(c, True)      #Getting perimeter of each contour

        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

        if len(approx) == 4:            #Selecting the contour with 4 corners/sides.

            NumberPlateCount = approx

            break

    #Masking all other parts, other than the number plate.

    masked = np.zeros(gray_img.shape,np.uint8)

    new_image = cv2.drawContours(masked,[NumberPlateCount],0,255,-1)

    new_image = cv2.bitwise_and(img,img,mask=masked)



    #Printing the recognized text as output.

    cv2.imshow("Preprocess 2 - Bilateral Filter", gray_img)    #Showing the preprocessed image.

    cv2.imshow("Preprocess 1 - Grayscale Conversion", gray_img)        #Show modification.
    cv2.imshow("Preprocess 3 - Canny Edges", c_edge)        #Showing the preprocessed image.
    cv2.imshow("Top 30 Contours", im2)  # Show the top 30 contours.

    cv2.imshow("4 - Final_Image", new_image)  # The final image showing only the number plate.

    cv2.waitKey(0)


if __name__ == '__main__':

    main()