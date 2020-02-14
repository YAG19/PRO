import cv2
import os
import time


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

url="D:\\WAL\\"
list1=(os.listdir(url))
for i in list1:
    img = cv2.imread("D:\\WAL\\"+i)
    #img1= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img1=img
    faces=face_cascade.detectMultiScale(img1,
    scaleFactor=1.05,minNeighbors=5)
    for x,y,w,h in faces:
        img1=cv2.rectangle(img1, (x,y) ,(x+w,y+h) , (0,255,0),3)
    print(faces)
    print(type(faces))

    resized=cv2.resize(img1 ,(int(img1.shape[1]/3),int(img1.shape[0]/3)))
   #resized=cv2.resize(img1 ,(500,700))

    cv2.imshow("L", resized)
    cv2.waitKey(1)
    time.sleep(2)
    cv2.destroyAllWindows()


#print(check)
#check,frame= video.read()
#print(frame)

#cv2.imshow("caturing",frame)
#video.release()

