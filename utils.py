from typing import List

import cv2
import numpy as np


def inaccuracy(cx: int, cy: int, accuracy=5) -> List:
    """
    This method make +- range from centroid coordinates
    :param cx: x-centroid
    :param cy: y-centroid
    :param accuracy: +- step
    :return: List(list, list), where list=c+-accuracy
    """
    cx_range = list(np.arange(cx - accuracy, cx + accuracy + 1))
    cy_range = list(np.arange(cy - accuracy, cy + accuracy + 1))
    return [cx_range, cy_range]


def make_gray_blur(frame: np.ndarray) -> np.ndarray:
    """
    This method transform frame into grayscale and then applies gaussian blur.
    :param frame: just frame of video stream
    :return: processed frame
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    return frame_blur