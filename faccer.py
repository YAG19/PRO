import os
import numpy as np
from PIL import Image
import cv2
import  pickle
url=os.getcwd()
base = (os.path.dirname(os.path.abspath(__file__)))
image_dir= os.path.join(url , "imgre")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_id={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
             path = os.path.join(root,file)
             label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
             #print(label,path)
             if not label in label_id:
                 label_id[label]=current_id
                 current_id+=1
             id_=label_id[label]
             #y_labels.append(label)
             #x_train.append(path)

             pil_image = Image.open(path).convert("L")
             image_array = np.array(pil_image,"uint8")
             #print(image_array)

             faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.05, minNeighbors=5)
             for x, y, w, h in faces:
                roi = image_array[y:y + h, x:x + w]
                x_train.append(roi)
                y_labels.append(id_)
print(y_labels,x_train)
with open("labels.pickle",'wb') as f:
    pickle.dump(label_id,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")



