CUDA_VISIBLE_DEVICES=0

import csv
import numpy as np
import cv2
import glob
from tensorflow.keras.models import load_model
from facial_analysis import FacialImageProcessing
imgProcessing=FacialImageProcessing(False)


INPUT_SIZE = (224, 224)
model=load_model('../models/affectnet_emotions/mobilenet_7.h5')
model.summary()


idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}


for fpath in glob.glob('/home/ubuntu/emotion_images/*.jpg'):
    frame_bgr=cv2.imread(fpath)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bounding_boxes, points = imgProcessing.detect_faces(frame)
    points = points.T
    for bbox,p in zip(bounding_boxes, points):
        box = bbox.astype(np.int)
        x1,y1,x2,y2=box[0:4]    
        face_img=frame[y1:y2,x1:x2,:]
        
        face_img=cv2.resize(face_img,INPUT_SIZE)
        inp=face_img.astype(np.float32)
        inp[..., 0] -= 103.939
        inp[..., 1] -= 116.779
        inp[..., 2] -= 123.68
        inp = np.expand_dims(inp, axis=0)
        scores=model.predict(inp)[0]

        max_score = idx_to_class[np.argmax(scores)]
        with open('emotion_labels.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([str(fpath), str(max_score)])