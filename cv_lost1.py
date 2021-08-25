import cv2
import numpy as np
from typing import List
from collections import defaultdict, namedtuple


def inaccuracy(cx: int, cy: int, accuracy=5) -> List:
    cx_range = list(np.arange(cx - accuracy, cx + accuracy + 1))
    cy_range = list(np.arange(cy - accuracy, cy + accuracy + 1))
    return [cx_range, cy_range]


def make_gray_blur(frame: np.ndarray) -> np.ndarray:
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
    return frame_blur


class DetectedObj:
    """

    """

    def __init__(self):
        self.coordinates = []
        self.start_frame = 0
        self.frame_count = 0
        self.abandoned_flag = False
        self.end_flad = False
        self.drop_counter = 0


class AbandonedDetection:
    """
    This class contains data about background frame and methods to detect
    abandoned objects on new frames
    """
    frame_num = 0
    obj_detected_dict = defaultdict(DetectedObj)

    def __init__(self, frame: np.ndarray):
        """
        Initialisation function that receive some frame to make it start
        background. It needs to find difference between frames
        :param frame: need to be cv2.read()[1] to correct processing
        """
        self.background = make_gray_blur(frame)

    def get_object(self, cnts, n: int):
        """

        :param cnts: contours of diff objects
        :param n: frame number
        :return:
        """
        cx, cy = 0, 0
        tmp_list = []

        for i, c in enumerate(cnts):
            moment = cv2.moments(cnts[i])
            if moment['m00'] == 0:
                pass
            else:
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])

            self.obj_detected_dict[(cx, cy)] = DetectedObj()

            if cv2.contourArea(c) > 400:
                (x, y, w, h) = cv2.boundingRect(c)
                # tmp_list.append((x, y, x+w, y+h))
                self.obj_detected_dict[(cx, cy)].coordinates = (x, y, x+w, y+h)
                self.obj_detected_dict[cx, cy].start_frame = n

                f = False

                inaccuracy_range = inaccuracy(cx, cy)
                for i in inaccuracy_range[0]:
                    for j in inaccuracy_range[1]:
                        if (i, j) in self.obj_detected_dict:
                            self.obj_detected_dict[(i, j)].frame_count += 1
                            cx, cy = i, j
                            f = True
                            break
                    if f:
                        break

                if self.obj_detected_dict[(cx, cy)].frame_count > 100:
                    self.obj_detected_dict[cx, cy].abandoned_flag = True

                self.drop_missed_box(cx, cy, n)

    def drop_missed_box(self, cx: int, cy: int, n_frame):
        start = self.obj_detected_dict[cx, cy].start_frame
        n_frames = self.obj_detected_dict[cx, cy].frame_count
        self.obj_detected_dict[cx, cy].end_flag = False

        if start + n_frames < n_frame:
            self.obj_detected_dict[cx, cy].end_flag = True

        if self.obj_detected_dict[cx, cy].end_flag:
            self.obj_detected_dict[cx, cy].drop_counter += 1

        if self.obj_detected_dict[cx, cy].drop_counter > 500:
            self.obj_detected_dict.pop([cx, cy])

    def find_difference(self, frame: np.ndarray) -> namedtuple:
        """
        This function find difference between background and new frame. After that it push
        difference to another methods to take boxes coordinates.
        :param frame: need to be cv2.read()[1] to correct processing
        :return: namedtuple(coordinates=(x1, y1, x2, y2), num_frames=n}
        """
        self.frame_num += 1
        frame = make_gray_blur(frame)
        frame_difference = cv2.absdiff(self.background, frame)

        edged = cv2.Canny(frame_difference, 30, 50)
        kernel = np.ones((8, 8), np.uint8)
        thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.get_object(cnts, self.frame_num)

        return self.obj_detected_dict
