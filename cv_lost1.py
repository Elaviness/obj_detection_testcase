from typing import Dict
from collections import defaultdict

import cv2
import numpy as np

from utils import inaccuracy, make_gray_blur


class DetectedObj:
    """
    This class only contains attributes for detected object.
    :coordinates x1,y1,x2,y2 box-coordinates
    :start_frame the frame from which object box appears
    :frame_count how many frames box are exist
    :abandoned_flag boolean flag, is object abandoned?
    :end_flag boolean flag, is object outside frame?
    :drop_counter how many frames object are oustide frame
    """

    def __init__(self):
        self.coordinates = []
        self.start_frame = 0
        self.frame_count = 0
        self.abandoned_flag = False
        self.end_flag = False
        self.drop_counter = 0


class AbandonedDetection:
    """
    This class contains data about background frame and methods to detect
    abandoned objects on new frames
    """
    _frame_num = 0
    _obj_detected_dict = defaultdict(DetectedObj)

    def __init__(self, frame: np.ndarray) -> None:
        """
        Initialisation function that receive some frame to make it start
        background. It needs to find difference between frames
        :param frame: need to be cv2.read()[1] to correct processing
        """
        self.background = make_gray_blur(frame)

    def _get_object(self, cnts, n: int) -> None:
        """
        This function work with list of contours of difference and compute centroid-coordinates,
        count repeats and call methods to lost boxes.
        :param cnts: contours of diff objects
        :param n: frame number
        :return:
        """
        cx, cy = 0, 0

        for i, c in enumerate(cnts):
            moment = cv2.moments(cnts[i])
            if moment['m00'] == 0:
                pass
            else:
                cx = int(moment['m10'] / moment['m00'])
                cy = int(moment['m01'] / moment['m00'])

            self._obj_detected_dict[(cx, cy)] = DetectedObj()

            if cv2.contourArea(c) > 400:
                (x, y, w, h) = cv2.boundingRect(c)
                self._obj_detected_dict[(cx, cy)].coordinates = (x, y, x + w, y + h)
                self._obj_detected_dict[cx, cy].start_frame = n

                f = False

                inaccuracy_range = inaccuracy(cx, cy)
                for j in inaccuracy_range[0]:
                    for k in inaccuracy_range[1]:
                        if (j, k) in self._obj_detected_dict:
                            self._obj_detected_dict[(j, k)].frame_count += 1
                            cx, cy = j, k
                            f = True
                            break
                    if f:
                        break

                if self._obj_detected_dict[(cx, cy)].frame_count > 100:
                    self._obj_detected_dict[cx, cy].abandoned_flag = True

                self._drop_missed_box(cx, cy, n)

    def _drop_missed_box(self, cx: int, cy: int, n_frame) -> None:
        """
        This function check if object on frame and how long it's outside.
        :param cx: x-centroid
        :param cy: y-centroid
        :param n_frame: current frame number
        :return: None
        """
        start = self._obj_detected_dict[cx, cy].start_frame
        n_frames = self._obj_detected_dict[cx, cy].frame_count
        self._obj_detected_dict[cx, cy].end_flag = False

        if start + n_frames < n_frame:
            self._obj_detected_dict[cx, cy].end_flag = True

        if self._obj_detected_dict[cx, cy].end_flag:
            self._obj_detected_dict[cx, cy].drop_counter += 1

        if self._obj_detected_dict[cx, cy].drop_counter > 500:
            self._obj_detected_dict.pop([cx, cy])

    def find_difference(self, frame: np.ndarray) -> Dict[tuple, DetectedObj]:
        """
        This function find difference between background and new frame. After that it push
        difference to another methods to take boxes coordinates.
        :param frame: need to be cv2.read()[1] to correct processing
        :return: dict where key - ( cx,cy) box-centroid coordinates, value -- class object DetectedObj
        """
        self._frame_num += 1
        frame = make_gray_blur(frame)
        frame_difference = cv2.absdiff(self.background, frame)

        edged = cv2.Canny(frame_difference, 30, 50)
        kernel = np.ones((8, 8), np.uint8)
        thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._get_object(cnts, self._frame_num)

        return self._obj_detected_dict
