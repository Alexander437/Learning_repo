
try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")

import time
import _thread as thread
import threading
import atexit
import sys
import termios
import contextlib

import imutils
import RPi.GPIO as GPIO
from adafruit_motorkit import MotorKit

from track_class import VideoUtils

### User Parameters ###

MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = True

MAX_STEPS_X = 30
MAX_STEPS_Y = 15

RELAY_PIN = 'LCD_TE'

#######################

class Turret(object):
    """
    Class used for turret control.
    """
    def __init__(self):

        # create a default object, no changes to I2C address or frequency
        self.kit = MotorKit()
        atexit.register(self.__turn_off_motors)
        self.current_x_steps = 0
        self.current_y_steps = 0

        # Relay
        #GPIO's mode is TEGRA_SOC with Motorkit
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    def run(self, camera_port=0, show_video=True):
        """
        Uses the camera to move the turret. OpenCV ust be configured to use this.
        :return:
        """
        v = VideoUtils()
        v.run_video(self.__move_axis, self.intaractive, camera_port=camera_port, show_video=show_video)

    def __move_axis(self, contour, frame):
        (v_h, v_w) = frame.shape[:2]
        (x, y, w, h) = contour

        # find height
        target_steps_x = (2*MAX_STEPS_X * (x + w / 2) / v_w) - MAX_STEPS_X
        target_steps_y = (2*MAX_STEPS_Y*(y+h/2) / v_h) - MAX_STEPS_Y

        print ("x: %s, y: %s" % (str(target_steps_x), str(target_steps_y)))
        print ("current x: %s, current y: %s" % (str(self.current_x_steps), str(self.current_y_steps)))

        t_x = threading.Thread()
        t_y = threading.Thread()
        t_fire = threading.Thread()

        # move x
        if (target_steps_x - self.current_x_steps) > 0:
            self.current_x_steps += 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=lambda: self.kit.stepper1.onestep(direction=1, style=1))
            else:
                t_x = threading.Thread(target=lambda: self.kit.stepper1.onestep(direction=0, style=1))
        elif (target_steps_x - self.current_x_steps) < 0:
            self.current_x_steps -= 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=lambda: self.kit.stepper1.onestep(direction=0, style=1))
            else:
                t_x = threading.Thread(target=lambda: self.kit.stepper1.onestep(direction=1, style=1))

        # move y
        if (target_steps_y - self.current_y_steps) > 0:
            self.current_y_steps += 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=lambda: self.kit.stepper2.onestep(direction=0, style=1))
            else:
                t_y = threading.Thread(target=lambda: self.kit.stepper2.onestep(direction=1, style=1))
        elif (target_steps_y - self.current_y_steps) < 0:
            self.current_y_steps -= 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=lambda: self.kit.stepper2.onestep(direction=1, style=1))
            else:
                t_y = threading.Thread(target=lambda: self.kit.stepper2.onestep(direction=0, style=1))

        # fire if necessary
        if True:
            if abs(target_steps_y - self.current_y_steps) <= 2 and abs(target_steps_x - self.current_x_steps) <= 2:
                t_fire = threading.Thread(target=self.fire)

        t_x.start()
        t_y.start()
        t_fire.start()

        t_x.join()
        t_y.join()
        t_fire.join()

    def interactive(self, key):
        """
        Starts an interactive session. Key presses determine movement.
        :return:
        """
        if key == 119:
            if MOTOR_Y_REVERSED:
                self.kit.stepper2.onestep(direction=1, style=1)
            else:
                self.kit.stepper2.onestep(direction=0, style=1)
        elif key == 115:
            if MOTOR_Y_REVERSED:
                self.kit.stepper2.onestep(direction=0, style=1)
            else:
                self.kit.stepper2.onestep(direction=1, style=1)
        elif key == 97:
            if MOTOR_X_REVERSED:
                self.kit.stepper1.onestep(direction=0, style=1)
            else:
                self.kit.stepper1.onestep(direction=1, style=1)
        elif key == 100:
            if MOTOR_X_REVERSED:
                self.kit.stepper1.onestep(direction=1, style=1)
            else:
                self.kit.stepper1.onestep(direction=0, style=1)
        elif (key==13) or (key==32):
            self.fire()

    @staticmethod
    def fire():
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(0.01)
        GPIO.output(RELAY_PIN, GPIO.LOW)
    


    def __turn_off_motors(self):
        """
        Recommended for auto-disabling motors on shutdown!
        :return:
        """
        self.kit.stepper1.release()
        self.kit.stepper2.release()


if __name__ == "__main__":

    t = Turret()
    t.run(camera_port=0, show_video=True)


