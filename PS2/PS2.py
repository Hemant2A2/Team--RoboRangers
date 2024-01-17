import gym
import os
import time as t
import LaRoboLiga24
import cv2
import pybullet as p

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
    'cam_dist'       : 30,
    'cam_yaw'        : 20,
    'cam_pitch'      : -130,
    'cam_target_pos' : [0,4,0]
})


os.chdir(os.path.dirname(os.getcwd()))
env = gym.make('LaRoboLiga24',
    arena = "arena2",
    car_location=CAR_LOCATION,
    ball_location=BALLS_LOCATION_BONOUS,  # toggle this to BALLS_LOCATION_BONOUS to load bonous arena
    humanoid_location=HUMANOIDS_LOCATION,
    visual_cam_settings=VISUAL_CAM_SETTINGS
)

"""
CODE AFTER THIS
"""

while True:
    img = env.get_image(cam_height=0, dims=[512, 512])
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
