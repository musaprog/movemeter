
import cv2

def load_image(fn):
    return cv2.imread(fn, -1)
