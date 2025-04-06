from botreceiver import BotReceiver
from segmentation_model import Segmentation_modelXL, Segmentation_model,Mask2FormerSegmentor
from global_planner import GPSRouteTracker
import osmnx as ox
from perspective_transform import PerspectiveTransformerGlobal
import numpy as np
import time
import matplotlib.pyplot as plt
from helper_func import rotation_planner, ImcrementatlVisualizer, visualize_trajectory_with_opencv
from bot_controller import BotController
# from inf import ImageSegmenter
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from collections import deque
from botpublisher import ROSPublisher
import time 
from PIL import Image
from motion_planner.dwa import DWA

#Substituting the dictionary values from UMD's original code
def create_dwa_observation(current_utm, orientation, goal, start_utm, linear_velocity, angular_velocity, bev_mask):
    obs = {
        "goal": (goal[0] - start_utm[0], goal[1] - start_utm[1]),  # Goal relative to start UTM
        "pose": [current_utm[0], current_utm[1], orientation],  # Pose relative to start UTM
        "velocity": [linear_velocity, angular_velocity],  # Linear and angular velocity
        "bev_mask": bev_mask  # Obstacle map from BEV segmentation
    }
    return obs

#gps collection and processing 
# global_counter =  0             #
#image_height = 576
#image_width = 1024
base_url = "http://localhost:8000"
node = 0                                            #initial node counter
receiver = BotReceiver(base_url)                    
ptransform = PerspectiveTransformerGlobal()
Segmentationmodel = Mask2FormerSegmentor()
resolution = 0.00539
planner = DWA()
destination_gps = (38.827620, -77.305884)
start_gps, start_orient = receiver.moving_average(10)
print(start_gps, "start_gps")
# vlts = ImageSegmenter("/home/robotixx/Amit/er/configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py", "/home/robotixx/Amit/er/checkpoin/vltseg_checkpoint_cityscapes_1.pth", 'cuda', {"load_from": '/home/robotixx/Amit/er/checkpoin/vltseg_checkpoint_cityscapes_1.pth'})
tracker = GPSRouteTracker(start_gps, destination_gps)
start_utm = tracker.gps_to_utm(start_gps[0],start_gps[1])
controller = BotController(base_url)
visualizer = ImcrementatlVisualizer(gps_tracker=tracker)
ros_publisher = ROSPublisher(start_utm)
route = tracker.get_route()
#direction = "STRAIGHT"
#entered_node = False
#turn_int = 1 
prev_time = 0.0
prev_img_timestamp = 0.0
#gps_buffer = []
#best_v_out, best_omega_out = 0,0
start_time = time.time()
prev_gps = 0 
action = [None,None]
#dummy data
import os   
import itertools
image_dir = "/home/robotixx/Sameep/frodobot_workspace/src/localization/src/robot_navigation/Input/images"
image_list = sorted(os.listdir(image_dir))  # Get a sorted list of image names
image_cycle = itertools.cycle(image_list)  
while True:
    data = receiver.fetch_bot_data()
    robot_pose,data_orientation, linear_velocity,angular_velocity = ros_publisher.parse_and_publish(data,tracker)
    # image = receiver.parse_image(camera="front")
    image_name = next(image_cycle)
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    current_utm, current_orientation =ros_publisher.get_latest_data()
    # print("the current orientation is ", current_orientation, data_orientation)
    ros_publisher.publish_image(np.array(image))

    if current_utm != None:
        # print(current_utm, " this is the current utm")
        distance, projected_distance, bearing = tracker.get_distance_to_next_utm(current_utm, start_utm,next_node_index= node) 
        error = (bearing - current_orientation+ np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
        # if abs(error) > 0.87:
        #     print(current_orientation, "there is error here")
        #     rotation_planner(error, current_orientation, bearing, controller, ros_publisher)
        print(current_orientation,"current_orientation")
        if distance  <= 6 :
            node = node+1
        #This condition is a fail-safe for when we don't receive data from api but only receive Image from the data.
        if linear_velocity == None and action[0] is not None:
            linear_velocity, angular_velocity = action[0],action[1]
        seg_mask = Segmentationmodel.predict(image)
        seg_mask = ptransform.inverse_perspective_mapping(seg_mask)
        visualizer.update(image =np.array(image),seg_mask=seg_mask)

        if not np.any(seg_mask == 1):
            print("No traversable areas (1s) in seg_mask. Skipping planner step.")
            continue
        goal = tracker.route_utm_coords[node]
        obs = create_dwa_observation(current_utm,current_orientation, goal , start_utm, linear_velocity, angular_velocity, seg_mask)
        action = planner.step(obs)
        print("action",action)
        if time.time()- start_time <=60:
            controller.send_control_command(.0, .0)
        else:
            controller.send_control_command(.0, .0)
        
        # visualize_trajectory(costmap, best_px,start_px, goal_px)
        # print(end_dist, "end_dist")
        time.sleep(.1)
visualizer.close()
"""
Average of mags1 -177.2872340425532
Average of mags2 905.5425531914893
Max of mags1 618.2872340425532
Max of mags2 731.4574468085107

"""