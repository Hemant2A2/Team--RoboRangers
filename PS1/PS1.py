import gym
import os
import time as t
import LaRoboLiga24
import cv2
import pybullet as p
import numpy as np
import math

CAR_LOCATION = [-25.5,0,1.5]
ROI_vertices = [(0,0),(0,450),(512,450),(512,0)]
ROI_vertices= np.array([ROI_vertices],np.int32)


VISUAL_CAM_SETTINGS = dict({
    'cam_dist'       : 40,
    'cam_yaw'        : 0,
    'cam_pitch'      : -110,
    'cam_target_pos' : [0,4,0]
})


os.chdir(os.path.dirname(os.getcwd()))
env = gym.make('LaRoboLiga24',
    arena = "arena1",
    car_location=CAR_LOCATION,
    visual_cam_settings=VISUAL_CAM_SETTINGS
)

"""
CODE AFTER THIS
"""

def ROI(img):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, ROI_vertices, match_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    #blurred = cv2.GaussianBlur(masked_img, (5, 5), 0)
    #edges = cv2.Canny(blurred, 50, 150)
    _, thresh = cv2.threshold(masked_img, 127, 255, 0)
    contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_center(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([160,50,50])
    upper_red = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(image,image, mask= mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return ROI(edges)

def control(mode):
    vel = 3
    if (mode == "S"):
        vel = 8
        env.move([[vel, vel], [vel, vel]])
    elif (mode == "R"):
        env.move([[0,0],[0,0]])
        env.move([[-vel, vel], [-vel, vel]])
    elif (mode == "L"):
        env.move([[0,0],[0,0]])
        env.move([[vel, -vel], [vel, -vel]])

def move(contours):
    if contours:
        cnt = max(contours, key= cv2.contourArea)

        _,cols = img.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        left_y = int((-x*vy/vx) + y)
        right_y = int(((cols-x)*vy/vx)+y)
        m = vy/vx

        cv2.line(img,(int(cols-1),right_y),(0,left_y),(0,255,0),2)
        cv2.line(img, (256, 512), (256, 0), (0,0,255), 2)

        offset = x - 256
        print(m, "\t OFFSET: ", offset)

        #Straight
        if  (-25 < offset < 25):
            control("S")
        #Right
        elif offset < -25:
            control("R")
        #Left
        elif offset > 25:
            control("L")

# def pid_controller(error, preError , integral):
#     Kp = 0.1
#     Ki = 0.01
#     Kd = 0.05

#     proportional = Kp * error
#     integral += Ki * error
#     derivative = Kd * (error - preError)

#     return proportional + integral + derivative


while True:
    img = env.get_image(cam_height=0, dims=[512, 512])
    contours = detect_center(img)
    
    move(contours)
    cv2.imshow("image", img)

    ## Manual control code ---------------------------------------------------------------------------------------------------------------------------------------
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
        

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
t.sleep(10)
env.close()