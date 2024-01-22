import gym
import os
import time as t
import LaRoboLiga24
import cv2
import pybullet as p
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

# def ROI(img, vertices):
#     mask = np.zeros_like(img)
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_img = cv2.bitwise_and(img, mask)
#     blurred = cv2.GaussianBlur(masked_img, (5, 5), 0)
#     cropped_img = cv2.Canny(blurred, 50, 150)
#     return cropped_img


def masking(image , lower_lim , upper_lim):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_lim, upper_lim)
    res = cv2.bitwise_and(image,image, mask= mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurred, 100, 200)
    return canny


def detect_yellow(image):
    lower_lim = np.array([20,50,50])
    upper_lim = np.array([40,255,255])
    return masking(image , lower_lim, upper_lim)


def detect_red(image):
    lower_lim = np.array([0,50,50])
    upper_lim = np.array([9,255,255])
    return masking(image , lower_lim, upper_lim)


def detect_blue(image):
    lower_lim = np.array([110,50,50])
    upper_lim = np.array([130,255,255])
    return masking(image , lower_lim, upper_lim)


def detect_purple(image):
    lower_lim = np.array([130,50,50])
    upper_lim = np.array([160,255,255])
    return masking(image , lower_lim, upper_lim)


def backtrack(Movements):
    for movement in reversed(Movements):
        m , v , i = movement
        move(m,v)
        t.sleep(i)


def open():
    env.open_grip()


def close():
    env.close_grip()


def shoot():
    env.shoot(5000)


def stop():
    env.move(vels=[[0, 0], [0, 0]])


def move(mode='f', speed=3):
    if mode.lower() == "f":
        vel = [[speed, speed], [speed, speed]]
    elif mode.lower() == "b":
        vel = [[-speed, -speed], [-speed, -speed]]
    elif mode.lower() == "r":
        vel = [[speed, -speed], [speed, -speed]]
    elif mode.lower() == "l":
        vel = [[-speed, speed], [-speed, speed]]
    env.move(vels=vel)


def isBall(cnt):
    area = cv2.contourArea(cnt) if cv2.contourArea(cnt) != 0 else 1
    x, y, w, h = cv2.boundingRect(cnt)
    return (True if (1.1 < w * h / area < 1.5) else False) if (0.85 < w / h < 1.2) else False


def MoveHold(cnt):
    global Movements
    x, y, w, h = cv2.boundingRect(cnt)
    x = x + w / 2
    if x > 302:
        start = t.time()
        move('r', (310 - x) / 120)
        end = t.time()
        interval = end - start
        Movements.append(('l', (310-x)/120 ,interval ))
    elif x < 298:
        start = t.time()
        move('r', (290 - x) / 120)
        end = t.time()
        interval = end - start
        Movements.append(('l',(290-x)/120 , interval))
    else:
        start = t.time()
        move('f', 5)
        end = t.time()
        interval = end - start
        Movements.append(('b',5 , interval))
        area = cv2.contourArea(cnt)
        if area > 30000:
            global Holding
            global cam_height
            stop()
            close()
            stop()
            cam_height = 1
            Holding = True


def MoveShoot(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    x = x + w / 2
    if x > 302:
        move('r', (310 - x) / 120)
    elif x < 298:
        move('r', (290 - x) / 120)
    else:
        global Goal
        #stop(1)
        open()
        shoot()
        t.sleep(2)
        Goal = True



while True:
    Movements = []
    img = env.get_image(cam_height=cam_height, dims=[512, 512])
    #ROI_vertices = [(160,320),(160,190),(360,190),(360,320)]

    ### Search order:
    ### yellow ball , blue ball , red ball , purple ball
    ### issue - not able to detect blue goal post
    ### after holding the ball crop the image / increase the camera height to remove the part containing ball

    canny = detect_blue(img)
    #cropped_img = ROI(canny, np.array([ROI_vertices],np.int32))
    Holding = False
    Center = False
    Goal = False

    _ , thresh = cv2.threshold(canny, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        cnt = max(contours , key=cv2.contourArea)
        cv2.drawContours(img, [cnt], 0, (0,255,0), 2)
        # if not Holding:
        #     if isBall(cnt):
        #         MoveHold(cnt)
        # elif not Center:
        #     backtrack(Movements)
        # else:
        #     MoveShoot(cnt)


    cv2.imshow("image",img)
    ## Manual control code
    keys = p.getKeyboardEvents()
    rot = 3
    speed = 5
    #vel = [[0,0],[0,0]]

    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_WAS_TRIGGERED:
        env.move([[-rot, rot], [-rot, rot]])

    elif p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_WAS_TRIGGERED:
        env.move([[rot, -rot], [rot, -rot]])

    elif p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED:
        env.move([[speed,speed ], [speed, speed]])

    elif p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED:
        env.move([[-speed, -speed], [-speed, -speed]])

    elif p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
        env.move([[0, 0], [0, 0]])

    elif ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
        env.open_grip()

    elif ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
        env.close_grip()

    elif ord('s') in keys and keys[ord('s')] & p.KEY_WAS_TRIGGERED:
        env.shoot(5000)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

t.sleep(10)
env.close()
