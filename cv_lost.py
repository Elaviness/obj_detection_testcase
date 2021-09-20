import sys
import cv2
import cv_lost1 as detect


def main():
    path = 'maks_test.mp4'
    #path = sys.argv[1]
    cap = cv2.VideoCapture(path)
    _, frame = cap.read()
    detector = detect.AbandonedDetection(frame)
    while (cap.isOpened()):
        _, frame = cap.read()
        res = detector.find_difference(frame)
        for r in res:
            (x, y, w, h) = res[r].coordinates
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # (frame, 'C %s,%s' % (cx, cy), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f'{r}-{res[r].frame_count}-{res[r].drop_counter};', r,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


            with open ('res.csv', 'a') as f:
                f.write(f'{r}|{res[r].frame_count}|{res[r].coordinates}|{res[r].drop_counter};')
            print(f'{r}|{res[r].frame_count}|{res[r].coordinates}|{res[r].end_flag}|{res[r].drop_counter};')

        cv2.imshow("Test", frame)

        cv2.waitKey(100)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
