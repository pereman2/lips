import cv2
import dlib 
import numpy as np
import imutils
import argparse
from imutils import face_utils
from collections import OrderedDict

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    #("right_eyebrow", (17, 22)),
    #("left_eyebrow", (22, 27)),
    #("right_eye", (36, 42)),
    #("left_eye", (42, 48)),
    #("nose", (27, 35)),
    #("jaw", (0, 17))
])
mouth = ("mouth", (48, 68))
# face_utils not working so I copied the method here
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    overlay = image.copy()
    output = image.copy()
    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
        (168, 100, 168), (158, 163, 32),
        (163, 38, 32), (180, 42, 220)]
    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]
        # check if are supposed to draw the jawline
        if name == "jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)
        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)
    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to input image")
args = vars(ap.parse_args())
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

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
