import os
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt

def save_detections(image, detections, out_dir='predicted'):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    for i, (x, y, _, w, h) in enumerate(detections):
        crop = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(out_dir, f'detection_{i+1}.png'), crop)

def show_detections(image, detections, title="Detections"):
    clone = image.copy()
    for (x, y, _, w, h) in detections:
        cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title(title)    
    plt.axis('off')
    plt.show()

def apply_nms(detections, overlap=0.3):
    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    scores = np.array([score[0] for (_, _, score, _, _) in detections])
    return non_max_suppression(rects, probs=scores, overlapThresh=overlap)
