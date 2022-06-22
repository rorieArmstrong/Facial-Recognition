import os
import cv2
import pickle
import numpy as np
from  PIL import Image

face_cascade = cv2.CascadeClassifier('./frontalface_alt2.xml')
recogniser = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk('./training-images'):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id = label_ids[label]

            pil_img = Image.open(path).convert("L")
            img_array = np.array(pil_img, "uint8")

            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5)

            for x,y,w,h in faces:
                roi = img_array[y:y+h,x:x+h]
                x_train.append(roi)
                y_labels.append(id)

with open("labels.pickel","wb") as f:
    pickle.dump(label_ids, f)

recogniser.train(x_train, np.array(y_labels))
recogniser.save("trainer.yml")