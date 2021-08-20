import sys
import cv2
import numpy as np


def preproc(path="test3.mp4"):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('result.mp4', fourcc, 20.0, (1280, 720), True)

    f = cv2.VideoCapture(path)
    rval, firstframe = f.read()
    f.release()
    firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    firstframe_blur = cv2.GaussianBlur(firstframe_gray, (21, 21), 0)

    counter = 0
    frameno = 0
    comparecx = [0]
    sumarea = 0
    return_dict = {}

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == 0:
            break
        frameno = frameno + 1

        # -------------------------------------------------
        # Frame + BGR2GRAY + Gaussian Blur + show frame_blur
        # -------------------------------------------------
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)

        frame_difference = cv2.absdiff(firstframe_blur, frame_blur)


        edged = cv2.Canny(frame_difference, 30, 50)
        kernel = np.ones((8, 8), np.uint8)
        thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=3)

        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.putText(frame, '%s' % ('l'), (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        tmp_list = []

        for i, c in enumerate(cnts):
            M = cv2.moments(cnts[i])
            if M['m00'] == 0:
                pass
            else:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

            if cv2.contourArea(c) > 400:
                (x, y, w, h) = cv2.boundingRect(c)
                tmp_list.append((x,y,w,h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, 'C %s,%s' % (cx, cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                comparecx.append(cx)
                if comparecx[0] > 100 and comparecx[1] < 100:
                    counter = counter + 1
                    sumarea = sumarea + cv2.contourArea(c)
                comparecx.pop(0)

        return_dict[frameno] = tmp_list
        img = cv2.resize(frame, (1280, 720))
        out.write(img)
        cv2.imshow('Window', frame)

        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return return_dict

if __name__ == "__main__":
    video_path = sys.argv[1]
    result = preproc(video_path)
    print(result)
