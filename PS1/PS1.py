import gym
import os
import time as t
import LaRoboLiga24
import cv2
import pybullet as p
import numpy as np
import math

CAR_LOCATION = [-25.5,0,1.5]


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

def ROI(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    blurred = cv2.GaussianBlur(masked_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def detect_center(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(image,image, mask= mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# def draw_lines(img, lines, color=[0, 0, 255], thickness=3):
#     image = np.copy(img)
#     if lines is None:
#         return
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(image, (x1, y1), (x2, y2), color, thickness)
#     return image

# def pid_controller(error, previous_error, integral,Kp, Ki, Kd):
#     proportional = Kp * error
#     integral += Ki * error
#     derivative = Kd * (error - previous_error)

#     return proportional + integral + derivative

# Kp = 0.1
# Ki = 0.01
# Kd = 0.05

while True:
    img = env.get_image(cam_height=0, dims=[512, 512])
    height = img.shape[0]
    width = img.shape[1]
    edges = detect_center(img)

    ROI_vertices = [(0,0),(0,450),(512,450),(512,0)]
    final_img = ROI(edges, np.array([ROI_vertices],np.int32))

    ret, thresh = cv2.threshold(final_img, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, -1, (0,0,255), 3)
    if contours:
        cnt = max(contours, key= cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        # c , dim , theta = rect
        # x,y = c
        # w,h = dim
        rows,cols = img.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(img,(int(cols-1),int(righty)),(0,int(lefty)),(0,255,0),2)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
    cv2.imshow("image", img)

    #change the argument values if needed
    # lines = cv2.HoughLinesP(
    #     edges,
    #     rho=6,
    #     theta=np.pi / 180,
    #     threshold=160,
    #     lines=np.array([]),
    #     minLineLength=40,
    #     maxLineGap=25
    # )

    # left_line_x = []
    # left_line_y = []
    # right_line_x = []
    # right_line_y = []
 
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         slope = (y2 - y1) / (x2 - x1)
    #     if slope <= 0:
    #         left_line_x.extend([x1, x2])
    #         left_line_y.extend([y1, y2])
    #     else:
    #         right_line_x.extend([x1, x2])
    #         right_line_y.extend([y1, y2])
    #     min_y = int(height*(3/5)) #change this if needed
    #     max_y = int(height)

    #     if left_line_y and left_line_x and right_line_x and right_line_y is not None:

    #         poly_left = np.poly1d(np.polyfit(
    #             left_line_y,
    #             left_line_x,
    #             deg=1
    #         ))
    #         left_x_start = int(poly_left(max_y))
    #         left_x_end = int(poly_left(min_y))

    #         poly_right = np.poly1d(np.polyfit(
    #             right_line_y,
    #             right_line_x,
    #             deg=1
    #         ))
    #         right_x_start = int(poly_right(max_y))
    #         right_x_end = int(poly_right(min_y))

    #         middle_x_start = int((left_x_start + right_x_start)/2)
    #         middle_x_end = int((left_x_end + right_x_end)/2)

    #         line_image = draw_lines(
    #             img,
    #             [[
    #                 [left_x_start, max_y, left_x_end, min_y], #uncomment to draw the left boundary
    #                 [right_x_start, max_y, right_x_end, min_y], #uncomment to draw the right boundary
    #                 #[middle_x_start, max_y,middle_x_end,min_y],

    #             ]],
    #         )
    
    # cv2.imshow("Line of Motion", line_image)



    ## Manual control code
    ## Need to make this automatic
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
