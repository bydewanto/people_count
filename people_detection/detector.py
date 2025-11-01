import cv2
import numpy as np
import joblib
from skimage.transform import pyramid_gaussian
from .features import extract_features

class PepsDetector:
    def __init__(self, model_path, min_window=(64, 64), step=(4, 16), downscale=1.5):
        self.clf = joblib.load(model_path)
        self.min_window = min_window
        self.step = step
        self.downscale = downscale
        self.detections = []

    def _scan(self, image, scale):
        h, w, _ = image.shape
        x, y = 0, 0
        prev = 0

        while y < h:
            while x < w:
                window = image[y:y + self.min_window[1], x:x + self.min_window[0]]
                if window.shape[:2] != self.min_window[::-1]:
                    break

                features = extract_features(window)
                pred = self.clf.predict(features)
                score = self.clf.decision_function(features)

                if pred == 1:
                    if prev == 0:
                        x = max(0, x - self.min_window[0] // 3)
                    self.detections.append(
                        (x, y, score, 
                         int(self.min_window[0] * (self.downscale ** scale)), 
                         int(self.min_window[1] * (self.downscale ** scale)))
                    )
                    x += self.step[0]
                else:
                    x += self.min_window[0] // 2

                prev = pred
            x, y = 0, y + self.step[1]

    def detect(self, image):
        self.detections.clear()
        scale = 0
        r, g, b = cv2.split(image)
        for r_p, g_p, b_p in zip(
            pyramid_gaussian(r, downscale=self.downscale),
            pyramid_gaussian(g, downscale=self.downscale),
            pyramid_gaussian(b, downscale=self.downscale)
        ):
            merged = cv2.merge((r_p, g_p, b_p))
            if merged.shape[0] < self.min_window[1] or merged.shape[1] < self.min_window[0]:
                break
            self._scan((merged * 255).astype("uint8"), scale)
            scale += 1
        return self.detections
