
import cv2, math, time, numpy as np
from ultralytics import YOLO
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

with ReachyMini() as reachy_mini:

    # pose = create_head_pose(x= 0, y = 0 ,z=0,  roll=0, pitch=-30, yaw=0, mm = True , degrees = True)
    # reachy_mini.goto_target(head=pose, duration=1.0)
    

    # pose = create_head_pose(x= 0, y = 0 ,z=0,  roll=0, pitch=+30, yaw=0, mm = True , degrees = True)
    # reachy_mini.goto_target(head=pose, duration=1.0)




    # Return to neutral
    pose = create_head_pose() 
    reachy_mini.goto_target(antennas=[math.radians(0), math.radians(0)], duration=1.0)
    reachy_mini.goto_target(head=pose, duration=2.0)