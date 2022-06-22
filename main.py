import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('./frontalface_alt2.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #FACE IDENTIFICATION
        

        #SAVE FACE AS PNG
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        #DRAW RECT AROUND FACES
        color = (0,255,255)
        stroke = 2
        cv2.rectangle(frame, (x,y),(x+w,y+h),color,stroke)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()