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

def detect_road_edges(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0,0,190])
    upper_white = np.array([172,111,255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    res = cv2.bitwise_and(image,image, mask= mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges

def draw_lines(img, lines, color=[0, 255, 0], thickness=5):
    image = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


"""def detect_gripper(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 5, 50])
    upper_gray = np.array([179, 50, 255])
    mask = cv2.inRange(hsv , lower_gray , upper_gray)
    res = cv2.bitwise_and(image,image, mask= mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    return cnt"""

# Function to implement PID controller for smooth turning
integral = 0
def pid_controller(error, previous_error, integral,Kp, Ki, Kd):
    proportional = Kp * error
    integral += Ki * error
    derivative = Kd * (error - previous_error)

    return proportional + integral + derivative

# Set PID gains for the controller
Kp = 0.1
Ki = 0.01
Kd = 0.05
previous_error = -1
lateral_error = -1
# Load a car model in PyBullet (you need to have a car model file)
#car = p.loadURDF("/Users/hemantayuj/Desktop/AI/Robotics/LA-ROBO-LIGA-24/urdf/car/car.urdf", [0, 0, 0.1])


while True:
    img = env.get_image(cam_height=1, dims=[512, 512])
    height = img.shape[0]
    width = img.shape[1]
    edges = detect_road_edges(img)

    #change the argument values if needed
    lines = cv2.HoughLinesP(
        edges,
        rho=6,
        theta=np.pi / 180,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
 
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
        if slope <= 0:
            left_line_x.extend([x1, x2])
            left_line_y.extend([y1, y2])
        else:
            right_line_x.extend([x1, x2])
            right_line_y.extend([y1, y2])
        min_y = int(height*(2/5)) #change this if needed
        max_y = int(height)

        if left_line_y and left_line_x and right_line_x and right_line_y is not None:
            poly_left = np.poly1d(np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            ))

            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))

            poly_right = np.poly1d(np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            ))

            right_x_start = int(poly_right(max_y))
            right_x_end = int(poly_right(min_y))
            line_image = draw_lines(
                img,
                [[
                    [left_x_start, max_y, left_x_end, min_y],
                    [right_x_start, max_y, right_x_end, min_y],
                ]],
            )

    #contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #sorted(contours, key=cv2.contourArea)
    #cv2.drawContours(img,contours,-1,(0,255,0),2)


    """cnt = detect_gripper(img)
    cv2.drawContours(img,[cnt], 0 , (255,0,0),2)
    gripper = cv2.moments(cnt)
    Gx = int(gripper['m10']/gripper['m00'])
    Gy = int(gripper['m01']/gripper['m00'])"""

    """minL = 0
    maxL = 3
    minR = 500
    maxR = 515
    draw = [contours[0],contours[1]]
    left = 0
    right = 500
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00']:
            cx = int(M['m10']/M['m00'])
            if cx >= minL and cx <= maxL:
                if left < cx:
                    left = cx
                    draw[0] = cnt
            elif cx >= minR and cx <= maxR:
                if right < cx:
                    right = cx
                    draw[1] = cnt
    cv2.drawContours(img , [draw[0]], 0 , (0,255,0), 2)
    cv2.drawContours(img , [draw[1]], 0 , (0,255,0), 2)
    print(f"{left}, {right}")"""

    """lx =0
    rx =0
    left_boundary = contours[-1]
    left = cv2.moments(left_boundary)
    if left['m00']:
        lx = int(left['m10']/left['m00'])
        ly = int(left['m01']/left['m00'])

    right_boundary = contours[-3]
    right = cv2.moments(right_boundary)
    if right['m00']:
        rx = int(right['m10']/right['m00'])
        ry = int(right['m01']/right['m00'])

    print(f"{lx} , {rx}")

    cv2.drawContours(img,[left_boundary],0,(0,255,0),2)
    cv2.drawContours(img,[right_boundary],0,(0,255,0),2)"""
    #print(len(contours))

    """for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            cv2.drawContours(img , [cnt] , 0 , (0,255,0), 2)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),2)
            rows,cols = img.shape[:2]
            [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            if cols-1 >=0 :
                cv2.line(img,(cols-1,righty),(0,lefty),(0,0,255),2)"""

    
    
    
    """for contour in contours:
        if cv2.contourArea(contour) < 100:
            cv2.drawContours(img , [contour] , 0 , (0,255,0), 2)
            road = cv2.moments(contour)
            Cx = int(road['m10']/road['m00'])
            Cy = int(road['m01']/road['m00'])
            print(f"({Cx} , {Cy})...")
            #road_center = cv2.moments(contour)
            if road_center["m00"] != 0:
                road_center_x = int(road_center["m10"] / road_center["m00"])
                road_center_y = int(road_center["m01"]/road_center["m00"])"""
                #print(f"{center_x-road_center_x}")
                #print(f"{road_center_y} -- {Gy}")

    

    

    """# Assume the largest contour is the road boundary
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(img, [largest_contour], 0, (0, 255, 0), 2)

        # Calculate the lateral error (distance from the center of the road)
        center_x = img.shape[1] / 2
        road_center = cv2.moments(largest_contour)
        if road_center["m00"] != 0:
            road_center_x = int(road_center["m10"] / road_center["m00"])
            lateral_error = center_x - road_center_x

        if previous_error != -1:
            # Apply PID control to adjust steering angle
            steering_angle = pid_controller(lateral_error, previous_error,integral, Kp, Ki, Kd)

            # Update the car's steering angle in PyBullet (adjust as needed)
            p.setJointMotorControl2(car, jointIndex=0, controlMode=p.POSITION_CONTROL, targetPosition=steering_angle)

        # Store the current error for the next iteration
        previous_error = lateral_error"""

    # Display the processed frame
    cv2.imshow("road detection", line_image)


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

    k = cv2.waitKey(1)
    if k == ord('q'):
        break
t.sleep(10)
env.close()
