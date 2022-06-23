import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('./data/frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('./data/eye.xml')
smile_cascade = cv2.CascadeClassifier('./data/smile.xml')

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read("trainer.yml")

labels = {"person":1}
with open("labels.pickel", "rb") as f:
    old_labels = pickle.load(f)
    labels = {v:k for k,v in old_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    for(x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #FACE IDENTIFICATION
        id_, conf = recogniser.predict(gray)
        if conf>= 80:
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            cv2.putText(frame, name, (x,y+h+30), font, 1, (255,255,255),2,cv2.LINE_AA)


        #SAVE FACE AS PNG
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        #DRAW RECT AROUND FACES
        color = (0,255,255)
        stroke = 2
        cv2.rectangle(frame, (x,y),(x+w,y+h),color,stroke)

    for(x,y,w,h) in eyes:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #DRAW RECT AROUND EYES
        color = (255,0,0)
        stroke = 2
        cv2.rectangle(frame, (x,y),(x+w,y+h),color,stroke)

    for(x,y,w,h) in smile:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #DRAW RECT AROUND EYES
        color = (255,0,0)
        stroke = 2
        cv2.rectangle(frame, (x,y),(x+w,y+h),color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()