from typing import Dict, Optional
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

    def __init__(self, frame: Optional[np.ndarray] = None, roi=None) -> None:
        """
        Initialisation function that receive some frame to make it start
        background. It needs to find difference between frames
        :param frame: need to be np.ndarray to correct processing
        """
        self._frame_num = 0
        self._obj_detected_dict = defaultdict(DetectedObj)
        self.background = None
        self.roi = roi
        self.background_renew = 0

    def is_object_in_roi(self, obj_rect):
        roi_x1 = self.roi[0]
        roi_y1 = self.roi[1]
        roi_x2 = self.roi[0] + self.roi[2]
        roi_y2 = self.roi[1] + self.roi[3]

        c_x1 = obj_rect[0]
        c_y1 = obj_rect[1]
        c_x2 = obj_rect[0] + obj_rect[2]
        c_y2 = obj_rect[1] + obj_rect[3]

        is_in = not (roi_x1 > c_x2 or roi_x2 < c_x1 or roi_y1 > c_y2 or roi_y2 < c_y1)
        is_enough = False

        if is_in:
            left = max(roi_x1, c_x1)
            right = min(roi_x2, c_x2)
            bottom = min(roi_y2, c_y2)
            top = max(roi_y1, c_y1)

            roi_perimeter = 2 * (self.roi[2] + self.roi[3])
            cross_perimeter = 2 * (right - left + bottom - top)

            is_enough = cross_perimeter >= 0.25 * roi_perimeter

        return is_enough

    def create_dict_obj(self, key, coordinates, frame, frame_count):
        self._obj_detected_dict[key].coordinates = coordinates
        self._obj_detected_dict[key].start_frame = frame
        self._obj_detected_dict[key].frame_count = frame_count

    def find_nms(self):
        objs = list(self._obj_detected_dict.keys())
        length = len(objs)
        i = 0
        j = 1
        while i < length - 1:
            fst_box = self._obj_detected_dict[objs[i]]
            while j < length:
                snd_box = self._obj_detected_dict[objs[j]]
                boxes = [fst_box.coordinates, snd_box.coordinates]
                idx = cv2.dnn.NMSBoxes(bboxes=boxes,
                                       scores=np.ones(len(boxes)),
                                       score_threshold=0.8,
                                       nms_threshold=0.6).flatten()[0]
                if idx.size == 1:
                    if idx == 0:
                        centroids = objs[i]
                        to_delete = objs[j]
                        j -= 1
                    else:
                        centroids = objs[j]
                        to_delete = objs[i]
                        i -= 1
                        j -= 1
                    length -= 1
                    coordinates = boxes[idx]
                    start_frame = min(self._obj_detected_dict[fst_box].start_frame,
                                      self._obj_detected_dict[snd_box].start_frame)
                    frame_count = max(self._obj_detected_dict[fst_box].frame_count,
                                      self._obj_detected_dict[snd_box].frame_count)

                    self.create_dict_obj(centroids,
                                         coordinates,
                                         start_frame,
                                         frame_count)

                    self._obj_detected_dict.pop(to_delete)
                    objs.remove(to_delete)
                j += 1
            i += 1

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

            f = False
            inaccuracy_range = inaccuracy(cx, cy, 20)
            for j in inaccuracy_range[0]:
                for k in inaccuracy_range[1]:
                    if (j, k) in self._obj_detected_dict:
                        cx, cy = j, k
                        f = True
                        break
                if f:
                    break

            (x, y, w, h) = cv2.boundingRect(c)

            # for obj in self._obj_detected_dict:
            #     old_box = self._obj_detected_dict[obj].coordinates
            #     new_box = (x, y, x + w, y + h)
            #     boxes = [old_box, new_box]
            #     idx = cv2.dnn.NMSBoxes(bboxes=boxes,
            #                            scores=np.ones(len(boxes)),
            #                            score_threshold=0.8,
            #                            nms_threshold=0.2).flatten()[0]
            #     if idx.size == 1:
            #         if idx == 0:
            #             (cx, cy) = obj
            #         else:
            #             self.create_dict_obj((cx, cy),
            #                                  (x, y, x + w, y + h),
            #                                  self._obj_detected_dict[obj].start_frame,
            #                                  self._obj_detected_dict[obj].frame_count)
            #             self
            #             self._obj_detected_dict.pop(obj)
            #         break

            if self.roi:
                if self.is_object_in_roi((x, y, w, h)) and not self._obj_detected_dict[(cx, cy)].start_frame:
                    self.create_dict_obj((cx, cy), (x, y, x + w, y + h), n, 0)
            elif cv2.contourArea(c) > 500 and not self._obj_detected_dict[(cx, cy)].start_frame:
                self.create_dict_obj((cx, cy), (x, y, x + w, y + h), n, 0)

            if (cx, cy) in self._obj_detected_dict:
                self._obj_detected_dict[(cx, cy)].frame_count += 1
                if self._obj_detected_dict[(cx, cy)].frame_count > 100:
                    self._obj_detected_dict[(cx, cy)].abandoned_flag = True

        if self._frame_num % 100 == 0 or len(self._obj_detected_dict) > 4:
            self.find_nms()

        self._drop_missed_box(n)


    def _drop_missed_box(self, n_frame) -> None:
        """
        This function check if object on frame and how long it's outside.
        :param cx: x-centroid
        :param cy: y-centroid
        :param n_frame: current frame number
        :return: None
        """
        tmp = []
        for obj in self._obj_detected_dict:
            start = self._obj_detected_dict[obj].start_frame
            n_frames = self._obj_detected_dict[obj].frame_count
            # self._obj_detected_dict[(cx, cy)].end_flag = False

            if start + n_frames < n_frame:
                #     self._obj_detected_dict[(cx, cy)].end_flag = True
                #
                # if self._obj_detected_dict[(cx, cy)].end_flag:
                self._obj_detected_dict[obj].drop_counter += 1

            if self._obj_detected_dict[obj].drop_counter > 200 or not self._obj_detected_dict[obj].coordinates:
                tmp.append(obj)

        for coordinates in tmp:
            self._obj_detected_dict.pop(coordinates)


    def find_difference(self, frame: np.ndarray) -> Dict[tuple, DetectedObj]:
        """
        This function find difference between background and new frame. After that it push
        difference to another methods to take boxes coordinates.
        :param frame: need to be np.ndarray to correct processing
        :return: dict where key - ( cx,cy) box-centroid coordinates, value -- class object DetectedObj
        """

        if self.background is None:
            self.background = make_gray_blur(frame)
        self._frame_num += 1
        frame_gb = make_gray_blur(frame)
        frame_difference = cv2.absdiff(self.background, frame_gb)

        edged = cv2.Canny(frame_difference, 30, 50)
        kernel = np.ones((8, 8), np.uint8)
        thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts and not self._obj_detected_dict:
            self.background_renew += 1
            if self.background_renew >= 1000:
                self.background = make_gray_blur(frame)

        self._get_object(cnts, self._frame_num)

        return self._obj_detected_dict
