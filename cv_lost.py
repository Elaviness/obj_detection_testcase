import sys
import cv2
import numpy as np
from collections import defaultdict


def inaccuracy(cx: int, cy: int, accuracy=5):
    cx_range = list(np.arange(cx - accuracy, cx + accuracy + 1))
    cy_range = list(np.arange(cy - accuracy, cy + accuracy + 1))
    return [cx_range, cy_range]


class InputVideo:
    """
    This class contains data about incoming video, that
    will be processed
    """
    path: str = ''
    output_path: str = ''
    box_dict: dict = {}

    def __init__(self, path: str) -> None:
        self.path = path
        self.output_path = path.split('.')[0] + '_result.mp4'

        # TODO: Сделать обработку выходного пути более универсальной, а не только от корня рабочей директории

    @property
    def processing(self):
        cap = cv2.VideoCapture(self.path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, 20.0, (1280, 720), True)

        _, fst_frame = cap.read()
        firstframe_gray = cv2.cvtColor(fst_frame, cv2.COLOR_BGR2GRAY)
        firstframe_blur = cv2.GaussianBlur(firstframe_gray, (21, 21), 0)

        counter = 0
        frameno = 0
        comparecx = [0]
        sumarea = 0
        top_contour_dict = defaultdict(int)

        while cap.isOpened():
            ret, frame = cap.read()

            if ret == 0:
                break
            frameno = frameno + 1
            print(type(frame))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)

            frame_difference = cv2.absdiff(firstframe_blur, frame_blur)

            edged = cv2.Canny(frame_difference, 30, 50)
            kernel = np.ones((8, 8), np.uint8)
            thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)

            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.putText(frame, '%s' % 'l', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            tmp_list = []
            # consecutiveframe = 20

            track_temp = []
            track_master = []
            # track_temp2 = []

            obj_detected_dict = defaultdict(int)
            cx, cy = 0, 0

            for i, c in enumerate(cnts):
                moment = cv2.moments(cnts[i])
                if moment['m00'] == 0:
                    pass
                else:
                    cx = int(moment['m10'] / moment['m00'])
                    cy = int(moment['m01'] / moment['m00'])

                if cv2.contourArea(c) > 400:
                    (x, y, w, h) = cv2.boundingRect(c)
                    tmp_list.append((x, y, w, h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    cv2.putText(frame, 'C %s,%s' % (cx, cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    # sumcxcy = cx + cy
                    track_temp.append([(cx, cy), frameno])

                    f = False

                    track_master.append([(cx, cy), frameno])
                    inaccuracy_range = inaccuracy(cx, cy)
                    for i in inaccuracy_range[0]:
                        for j in inaccuracy_range[1]:
                            if (i, j) in top_contour_dict:
                                top_contour_dict[(i, j)] += 1
                                cx, cy = i, j
                                f = True
                                break
                        if f:
                            break

                    # countuniqueframe = set(j for i, j in track_master)
                    #
                    # if len(countuniqueframe) > consecutiveframe:
                    #     minframeno = min(j for i, j in track_master)
                    #     for i, j in track_master:
                    #         if j != minframeno:
                    #             track_temp2.append([i, j])
                    #
                    #     track_master = list(track_temp2)
                    #     track_temp2 = []
                    #
                    # countcxcy = Counter(i for i, j in track_master)
                    #
                    # for i, j in countcxcy.items():
                    #     if j >= consecutiveframe:
                    #         top_contour_dict[i] += 1

                    if top_contour_dict[(cx, cy)] > 100:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        cv2.putText(frame, '%s' % 'CheckObject', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2)
                        # TODO Бросать флаг найденыша/пуш уведомления об оставленном объекте
                        sumcxcy = cx + cy
                        obj_detected_dict[sumcxcy] = frameno

                # for i, j in obj_detected_dict.items():
                #     if frameno - obj_detected_dict[i] > 200:
                #         obj_detected_dict.pop(i)
                #
                #         top_contour_dict[i] = 0

                # cv2.imshow('Abandoned Object Detection', frame)

# TODO Если н-фреймов подряд объект из obj_detected отсутствует -- убрать его из всех словарей
                comparecx.append(cx)
                if comparecx[0] > 100 > comparecx[1]:
                    counter = counter + 1
                    sumarea = sumarea + cv2.contourArea(c)
                # comparecx.pop(0)

            self.box_dict[frameno] = tmp_list
            img = cv2.resize(frame, (1280, 720))
            out.write(img)

        cap.release()
        out.release()

        return self.box_dict


def main():
    # path = 'test5.mp4'
    path = sys.argv[1]
    cap = InputVideo(path)
    cap.processing
    cv2.destroyAllWindows()

    print(cap.box_dict)


if __name__ == '__main__':
    main()
