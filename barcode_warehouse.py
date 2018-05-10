import numpy as np
import cv2
from pyzbar.pyzbar import decode

cap = cv2.VideoCapture(0)

while True:

	ret, image = cap.read()

	cv2.namedWindow("Window", cv2.WINDOW_NORMAL)
	cv2.imshow("Window",image)

	#image=cv2.imread("barcode1.jpg")
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	gradX=cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1,dy=0,ksize=-1)
	gradY=cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0,dy=1,ksize=-1)
	gradient=cv2.subtract(gradX,gradY)
	gradient=cv2.convertScaleAbs(gradient)
#Now we are left with the region of the image with high horizontal gradient and low vertical gradient

	blurred = cv2.blur(gradient, (9, 9))
	(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	closed = cv2.erode(closed, None, iterations = 4)
	closed = cv2.dilate(closed, None, iterations = 4)
#This reduces the noise

	_, cnts, _= cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

	if (len(cnts)==0):
		continue

	c = sorted(cnts, key = cv2.contourArea, reverse = True)[0] 

	rect = cv2.minAreaRect(c)
	box = np.int0(cv2.boxPoints(rect))
#box contains the vertices of barcode region rectangle

	y1=np.min(box[:,0])
	y2=np.max(box[:,0])
	x1=np.min(box[:,1])
	x2=np.max(box[:,1])

	barcode=image[x1:x2,y1:y2] #barcode is extracted


	cv2.imwrite("/Users/krishna/Desktop/barcode.jpg",barcode)
	print(decode(barcode))
	cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()