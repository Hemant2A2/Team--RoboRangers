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
pre = 0
i = 0

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
### Team--RoboRangers/urdf/arena/robo_lega/materials ---> use the track.jpg image in drive here

def ROI(img):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, ROI_vertices, match_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    ret, thresh = cv2.threshold(masked_img, 127, 255, 0)
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


def control(mode, speed):    
    vel = abs(5 + speed)
    if (vel <= 1.25):
        vel = 2.6    
    print(speed)
    if (mode == "S"):  
        vel = 11
        env.move([[vel, vel], [vel, vel]])
    elif (mode == "R"):
        env.move([[(-vel), vel], [(-vel), vel]])  
        
    elif (mode == "L"):
        env.move([[vel, (-vel)], [vel, (-vel)]])
       

def move(contours, preError, integral):
    if contours:
        cnt = max(contours, key= cv2.contourArea)
        rows,cols = img.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        m = vy/vx
        new_y = 156
        new_x = (new_y - (y-m*x))/m
        offset = int((new_x - 256)/10)
        speed = pid(x, preError, integral)

        #Straight
        if  (-5 < offset < 5):
            speed = pid(x, preError, integral)    
            control("S",speed)
            
        #Right
        elif offset < -5:
            speed = pid(x, preError, integral)    
            control("R", speed)
        #Left
        elif offset > 5:
            speed = pid(x, preError, integral)    
            control("L", speed)
            

def pid(value, preError, integral):
    Kp = 0.0025
    Ki = 0.04
    Kd = 0.05
    error = (value - 256)
    preError += error 
    proportional = Kp * error
    integral += Ki * preError
    derivative = Kd * (error - preError)

    return proportional + integral + derivative


    
while True:
    img = env.get_image(cam_height=0, dims=[512, 512])
    contours = detect_center(img)

    move(contours, pre, i)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
t.sleep(10)
env.close()