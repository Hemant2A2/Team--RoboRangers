import gym
import os
import time as t
import LaRoboLiga24
import cv2
import numpy as np

CAR_LOCATION = [0,0,1.5]
cam_height = 0


BALLS_LOCATION = dict({
    'red': [7, 4, 1.5],
    'blue': [2, -6, 1.5],
    'yellow': [-6, -3, 1.5],
    'maroon': [-5, 9, 1.5]
})
BALLS_LOCATION_BONOUS = dict({
    'red': [9, 10, 1.5],
    'blue': [10, -8, 1.5],
    'yellow': [-10, 10, 1.5],
    'maroon': [-10, -9, 1.5]
})

HUMANOIDS_LOCATION = dict({
    'red': [11, 1.5, 1],
    'blue': [-11, -1.5, 1],
    'yellow': [-1.5, 11, 1],
    'maroon': [-1.5, -11, 1]
})

VISUAL_CAM_SETTINGS = dict({
    'cam_dist'       : 20,
    'cam_yaw'        : 20,
    'cam_pitch'      : -120,
    'cam_target_pos' : [0,4,0]
})


os.chdir(os.path.dirname(os.getcwd()))
env = gym.make('LaRoboLiga24',
    arena = "arena2",
    car_location=CAR_LOCATION,
    ball_location=BALLS_LOCATION,  # toggle this to BALLS_LOCATION_BONOUS to load bonous arena
    humanoid_location=HUMANOIDS_LOCATION,
    visual_cam_settings=VISUAL_CAM_SETTINGS
)

"""
CODE AFTER THIS
"""

def masking(image , lower_lim , upper_lim):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_lim, upper_lim)
    res = cv2.bitwise_and(image,image, mask= mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 50, 150)
    return canny


def detect_yellow(image):
    lower_lim = np.array([20,50,50], dtype = np.uint8)
    upper_lim = np.array([40,255,255], dtype = np.uint8)
    return masking(image , lower_lim, upper_lim)


def detect_red(image):
    lower_lim = np.array([0,50,50], dtype = np.uint8)
    upper_lim = np.array([9,255,255], dtype = np.uint8)
    return masking(image , lower_lim, upper_lim)


def detect_blue(image):
    lower_lim = np.array([110,50,50], dtype = np.uint8)
    upper_lim = np.array([130,255,255], dtype = np.uint8)
    return masking(image , lower_lim, upper_lim)


def detect_purple(image):
    lower_lim = np.array([130,50,50], dtype = np.uint8)
    upper_lim = np.array([160,255,255], dtype = np.uint8)
    return masking(image , lower_lim, upper_lim)


def backtrack(ball_no):
    if ball_no == 4:
        move('b',14,2)  ## for backtracking purple ball
    elif ball_no == 1:
        move('b',7,2)   ## for backtracking yellow ball
    elif ball_no == 2:
        move('b',9,2)   ## for backtracking red ball
    else:
        move('b',8,2)   ## for backtracking blue ball
    global Center
    Center = True


def open():
    env.open_grip()


def close():
    env.close_grip()


def shoot():
    env.shoot(1000)


def stop():
    env.move(vels=[[0, 0], [0, 0]])


def move(mode='f', speed=1.5 , interval = 0):
    if mode.lower() == "f":
        vel = [[speed, speed], [speed, speed]]
    elif mode.lower() == "b":
        vel = [[-speed, -speed], [-speed, -speed]]
    elif mode.lower() == "r":
        vel = [[speed, -speed], [speed, -speed]]
    elif mode.lower() == "l":
        vel = [[-speed, speed], [-speed, speed]]
    env.move(vels=vel)
    t.sleep(interval)


def isBall(cnt):
    area = cv2.contourArea(cnt) if cv2.contourArea(cnt) != 0 else 1
    x, y, w, h = cv2.boundingRect(cnt)
    return (True if (1.2 < w * h / area < 1.6) else False) if (0.85 < w / h < 1.3) else False

def MoveHold(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    x = x + w / 2
    if x > 265:
        move('r', (x - 260) / 120)
    elif x < 245:
        move('l', (250 - x) / 120)
    else:
        move('f', 5)
        area = cv2.contourArea(cnt)
        if area > 23000:
            global Holding
            global cam_height
            stop()
            close()
            stop()
            cam_height = 1.5
            Holding = True


def MoveShoot(cnt):
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(img,center,radius,(0,0,0),3)
    if x < 290 and x > 170:
        global Goal
        stop()
        open()
        shoot()
        t.sleep(2)
        Goal = True

############  INITIAL CONDITIONS  ##############
def initial():
    open()
    global Holding
    Holding = False
    global Center
    Center = False
    global Goal
    Goal = False
    global cam_height
    cam_height = 0
p = y = r = b = False
p_init = y_init = r_init = b_init = False
################################################

while True:
    img = env.get_image(cam_height=cam_height, dims=[512, 512])

    def Find(canny,ball_no):
        global Holding , Center , Goal, ball_location
        global p,y,r,b
        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if Holding:
            if not Center:
                backtrack(ball_no)
                if ball_no is 3:
                    move('r',4,2.7)
                    stop()
                    open()
                    shoot()
                    t.sleep(2)
                    Goal = True
                    b = True

        if contours:
            cnt = max(contours , key=cv2.contourArea)
            cv2.drawContours(img, [cnt], 0, (0,255,0), 2)
            area = cv2.contourArea(cnt)
            if not Holding:
                if isBall(cnt):
                    MoveHold(cnt)
                else:
                    if ball_no is 1:
                        move('l')
                    else:
                        move('r')
            else:
                if Goal:
                    if ball_no == 1:
                        y = True
                    elif ball_no == 2:
                        r = True
                    elif ball_no == 3:
                        b = True
                    else:
                        p = True
                elif area > 2000:
                    MoveShoot(cnt)
                else:
                    move('r')
        else:
            move('r')

    if not y:
        if not y_init:
            initial()
            y_init = True
        Find(detect_yellow(img),1)
    elif not r:
        if not r_init:
            initial()
            r_init = True
        Find(detect_red(img),2)
    elif not b:
        if not b_init:
            initial()
            b_init = True
        Find(detect_blue(img),3)
    elif not p:
        if not p_init:
            initial()
            p_init = True
        Find(detect_purple(img),4)
    else:
        break

    cv2.imshow("image",img)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

t.sleep(10)
env.close()
