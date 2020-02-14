import cv2
import os
import time


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")


img = cv2.imread("D:\\WAL\\abc (41)")
faces=face_cascade.detectMultiScale(img,scaleFactor=1.05,minNeighbors=5)
for x,y,w,h in faces:
    img=cv2.rectangle(img, (x,y) ,(x+w,y+h) , (0,255,0),3)
    
    id_,conf=recognizer.predict()


resized=cv2.resize(img ,(int(img.shape[1]/3),int(img.shape[0]/3)))
#resized=cv2.resize(img ,(500,700))

cv2.imshow("L", resized)
cv2.waitKey()

cv2.destroyAllWindows()