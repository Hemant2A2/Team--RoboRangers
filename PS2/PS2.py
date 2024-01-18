import gym
import os
import time as t
import LaRoboLiga24
import cv2
import pybullet as p
import numpy as np

CAR_LOCATION = [0,0,1.5]

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
    canny = cv2.Canny(gray, 100, 200)
    return canny

def yellow_ball(image):
    lower_lim = np.array([20,50,50])
    upper_lim = np.array([40,255,255])
    return masking(image , lower_lim, upper_lim)

# def yellow_goal(image):
#     lower_lim = np.array([20,50,50])
#     upper_lim = np.array([40,255,255])
#     return masking(image , lower_lim, upper_lim)

def red_ball(image):
    lower_lim = np.array([0,50,50])
    upper_lim = np.array([9,255,255])
    return masking(image , lower_lim, upper_lim)

# def red_goal(image):
#     lower_lim = np.array([0,50,50])
#     upper_lim = np.array([9,255,255])
#     return masking(image , lower_lim, upper_lim)

def blue_ball(image):
    lower_lim = np.array([90,50,50])
    upper_lim = np.array([128,255,255])
    return masking(image , lower_lim, upper_lim)

# def blue_goal(image):
#     lower_lim = np.array([90,50,50])
#     upper_lim = np.array([128,255,255])
#     return masking(image , lower_lim, upper_lim)

def purple_ball(image):
    lower_lim = np.array([130,50,50])
    upper_lim = np.array([160,255,255])
    return masking(image , lower_lim, upper_lim)

# def purple_goal(image):
#     lower_lim = np.array([130,50,50])
#     upper_lim = np.array([160,255,255])
#     return masking(image , lower_lim, upper_lim)

while True:
    img = env.get_image(cam_height=0, dims=[512, 512])

    # Search order:
    # yellow ball , blue ball , red ball , purple ball
    # issue - not able to detect blue goal post
    # after holding the ball crop the image to remove the part containing ball

    canny = blue_ball(img)
    ret, thresh = cv2.threshold(canny, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours , key=cv2.contourArea)
        cv2.drawContours(img, [cnt], 0, (0,255,0), 2)

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
        env.shoot()

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

t.sleep(10)
env.close()
