import cv2
import time

class SaveFrame():

    def __init__(self, camera_port=0):
        self.camera_port = camera_port
        self.mode = 0

    def __call__(self, label, delay=20):
        cap = cv2.VideoCapture(self.camera_port)
        count = 1
        count_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            path = '/home/pi/Learning_repo/images/' + label + '_' + str(count) + '.jpeg'
            if (self.mode%2 == 1):
                count_frame += 1
                if (count_frame%delay == 0):
                    cv2.imwrite(path, frame)
                    print('writing img: ', path)
                    count += 1

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)

            if key == ord('r'):
                self.mode += 1
                time.sleep(1)
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
