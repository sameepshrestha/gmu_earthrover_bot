from botreceiver import BotReceiver
from segmentation_model import Segmentation_modelXL, Segmentation_model,Mask2FormerSegmentor
from global_planner import GPSRouteTracker
import osmnx as ox
from perspective_transform import PerspectiveTransformer
from dwa_planner import DWAPlanner, pixel_to_metric, metric_to_pixel, local_goal_selection, label2name_all
import numpy as np
import time
import matplotlib.pyplot as plt
from helper_func import compute_velocity_from_rpms, test_all_heading_candidates, save_transformed_image, MovingAverage, ImcrementatlVisualizer, visualize_trajectory_with_opencv
from bot_controller import BotController
# from inf import ImageSegmenter
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import csv
from collections import deque
from botpublisher import ROSPublisher
import time 
#gps collection and processing 
global_counter =  0 
image_height = 576
image_width = 1024
base_url = "http://localhost:8000"
node = 0 
reveiver = BotReceiver(base_url)
ptransform = PerspectiveTransformer()
Segmentationmodel = Mask2FormerSegmentor()
planner = DWAPlanner()
start_gps, start_orientation  = reveiver.moving_average(10)
destination_gps = (38.827675, -77.305605)

# vlts = ImageSegmenter("/home/robotixx/Amit/er/configs/mask2former_evaclip_2xb8_5k_gta2cityscapes.py", "/home/robotixx/Amit/er/checkpoin/vltseg_checkpoint_cityscapes_1.pth", 'cuda', {"load_from": '/home/robotixx/Amit/er/checkpoin/vltseg_checkpoint_cityscapes_1.pth'})
 
tracker = GPSRouteTracker(start_gps, destination_gps)
start_utm = tracker.gps_to_utm(start_gps[0],start_gps[1])
controller = BotController(base_url)
visualizer = ImcrementatlVisualizer(gps_tracker=tracker)
ros_publisher = ROSPublisher(start_utm)
route = tracker.get_route()
# print(start_gps, start_orientation)

direction = "STRAIGHT"
entered_node = False
turn_int = 1 
# print("i am here")
resolution = 0.00539

prev_time = 0.0
prev_img_timestamp = 0.0
gps_buffer = []
best_v_out, best_omega_out = 0,0
start_time = time.time()
prev_gps = 0 

while True:
    data = reveiver.fetch_bot_data()
    if prev_time != data["timestamp"]:
        # print( "not _skipped",data)
        prev_time = data["timestamp"]
        rpm_data = data['rpms'][-1][0:4]
        gps_lat_avg =data["latitude"]
        gps_lon_avg = data["longitude"]
        orientation = ros_publisher.calculate_compass_angle(data["mags"][-1][1], data["mags"][-1][2])
        current_gps = (gps_lat_avg, gps_lon_avg)
        linear_velocity,angular_velocity = compute_velocity_from_rpms(rpm_data)
        utm_x, utm_y = tracker.gps_to_utm(gps_lat_avg,gps_lon_avg)
        # print(linear_velocity, angular_velocity,orientation , "odom topics publishing ")
        ros_publisher.publish_data(linear_velocity, angular_velocity, orientation, utm_x, utm_y, timestamps = [data["timestamp"],data['rpms'][-1][4], data["mags"][0][3]])
    else:
        # current_gps = (gps_lat_avg.get_average(), gps_lon_avg.get_average())
        # orientation = gps_orient_avg.get_average()
        linear_velocity,angular_velocity = best_v_out, best_omega_out
    # visualizer.update(gps_data=(current_gps))
    # print("skipped",data["timestamp"])
    image = reveiver.parse_image(camera="front")
    print(ros_publisher.get_latest_data(),"this shit is subsriber by the way")
    ros_publisher.publish_image(np.array(image))
    distance, bearing = tracker.get_distance_to_next(current_gps, next_node_index= node) 
    # projected_distance = tracker.get_projected_distance
    error = (orientation - bearing + 360) % 360
    # print(current_gps)
    if error > 180:
        error -= 360
    # print("the error in the angle is and the error distance , current_gps and route,node ",error, distance, current_gps, route, node)
    if distance  <=6 :
        node = node+1
        # print("NOOOOOOOOODE CHANGED")
        error = 0 
        # print("node changed")
    if abs(error) <= 20:
        direction = "STRAIGHT"
    elif error > 0:
        dorientationirection = "RIGHT"
    else:
        direction = "LEFT"
    # image = np.array(image)
    # seg_mask = inf.segment_image(image, "/home/kintou/Work/Robotixx/er/configs/mask2former_evaclip_2xb8_1k_frozen_gta2cityscapes.py", "/home/kintou/Work/Robotixx/er/checkpoin/vltseg_checkpoint_cityscapes_2.pth", 'cuda', {"load_from": '/home/kintou/Work/Robotixx/er/checkpoin/vltseg_checkpoint_cityscapes_2.pth'})
    seg_mask = Segmentationmodel.predict(image)
    # save_data(seg_mask, "segmentation")
    seg_mask = ptransform.inverse_perspective_mapping(seg_mask)
    origin = (seg_mask.shape[0]-1, seg_mask.shape[1]//2)
    start_px = (origin[0],origin[1])
    cols, rows = local_goal_selection(seg_mask,direction = direction, row_search=220)
    goal_px = (rows, cols)
    start_metric = pixel_to_metric(start_px[1], start_px[0], resolution, origin)  # (px, py)
    goal_metric = pixel_to_metric(goal_px[1], goal_px[0], resolution, origin) 
    current_state = [start_metric[0], start_metric[1], np.pi / 2, linear_velocity, angular_velocity] 
    best_v_out,best_omega_out,best_trajectory, best_px, costmap = planner.plan(current_state, goal_metric, seg_mask, label2name_all,resolution, origin)
    # save_data(costmap, "cost_map")    print(best_v, best_omega,"best_v, best_omega")
    # print(best_v_out, best_omega_out,"best_v, best_omega")
    # best_v,best_omega = normalize_velocity(best_v_out, best_omega_out )
    controller.send_control_command(best_v_out, best_omega_out)
    # if time.time()- start_time<=400:is is the orientation
    #     controller.send_control_command(0.15, 0)
    # else:
    #     controller.send_control_command(0, 0)
    # visualizer.update(image =np.array(image),seg_mask = Seg_mask,traj_data=visualize_trajectory_with_opencv(costmap, best_px,start_px, goal_px))
    # visualize_trajectory(costmap, best_px,start_px, goal_px)
    # print(end_dist, "end_dist")
    time.sleep(.01)
visualizer.close()
    
