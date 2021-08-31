import sys
import cv2
import cv_lost1 as detect


def main():
    # path = 'test5.mp4'
    path = sys.argv[1]
    cap = cv2.VideoCapture(path)
    _, frame = cap.read()
    detector = detect.AbandonedDetection(frame, (689, 100, 239, 346))
    while (cap.isOpened()):
        _, frame = cap.read()
        res = detector.find_difference(frame)
        print(res)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
