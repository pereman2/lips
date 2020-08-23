import argparse

import cv2
import dlib
import hydra
import imutils
import numpy as np
from imutils import face_utils

from omegaconf import DictConfig
from utils import visualize_facial_landmarks

mouth = ("mouth", (48, 68))

@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig) -> None:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(hydra.utils.to_absolute_path(cfg.shape_predictor))
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        if not rects:
            continue
        clone = image.copy()
        i, rect = 0, rects[0]
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        i, j = mouth[1]
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        # mouth image
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        output = visualize_facial_landmarks(image, shape)
        cv2.imshow("Image", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
