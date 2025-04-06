from botreceiver import BotReceiver
from segmentation_model import Segmentation_modelXL, Segmentation_model
from global_planner import GPSRouteTracker
import osmnx as ox
from perspective_transform import PerspectiveTransformer
from dwa_planner import DWAPlanner, pixel_to_metric, metric_to_pixel, local_goal_selection, label2name_all
import numpy as np
import time
import matplotlib.pyplot as plt
from helper_func import compute_velocity_from_rpms, save_data, normalize_velocity, save_transformed_image, MovingAverage, ImcrementatlVisualizer, visualize_trajectory_with_opencv
from bot_controller import BotController
# import inf
from botpublisher import ROSPublisher
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import csv
from collections import deque
from config import BASE_URL, IMAGE_HEIGHT, IMAGE_WIDTH, DESTINATION_GPS, RESOLUTION, MAX_BUFFER_SIZE


class RobotNavigationSystem:
    def __init__(self, destination_gps, resolution=0.00539):
        self.receiver = BotReceiver(BASE_URL)
        self.transformer = PerspectiveTransformer()
        self.segmentor = Segmentation_modelXL()
        self.planner = DWAPlanner()
        self.controller = BotController(BASE_URL)
        self.start_gps = self.receiver.moving_average(15)
        self.tracker = GPSRouteTracker(self.start_gps, self.destination_gps)
        self.visualizer = ImcrementatlVisualizer(gps_tracker=self.tracker)
        self.start_utm = self.tracker.gps_to_utm(self.start_gps)
        self.frodobot_publisher = ROSPublisher(start_utm=self.start_utm)
        # Initialize state variables
        self.node = 0
        self.prev_time = 0.0
        self.best_v_out = 0.0
        self.best_omega_out = 0.0

    def run(self, current_gps, orientation, linear_v, angular_v, image):
        """Update navigation parameters and generate control commands"""
        cuurent_x, current_y, orientation  =self.frodobot_publisher.publish_data()
        distance, bearing = self.tracker.get_distance_to_next()#add start_utm to x and y  , distance could be 0 or 1 either based or distance itself
        if bool(distance):  
           self.node += 1
        # Process image and plan path
        seg_mask = self.segmentor.predict(image)
        seg_mask = self.transformer.inverse_perspective_mapping(seg_mask)
        
        # Path planning
        origin = (seg_mask.shape[0]-1, seg_mask.shape[1]//2)
        start_px = (origin[0], origin[1])
        cols, rows = local_goal_selection(seg_mask, direction=self.direction, row_search=220)
        goal_px = (rows, cols)
        
        start_metric = pixel_to_metric(start_px[1], start_px[0], self.resolution, origin)
        goal_metric = pixel_to_metric(goal_px[1], goal_px[0], self.resolution, origin)
        
        current_state = [start_metric[0], start_metric[1], np.pi/2, linear_v, angular_v]
        v_out, omega_out, trajectory, px, costmap = self.planner.plan(
            current_state, goal_metric, seg_mask, label2name_all, self.resolution, origin
        )
        
        return normalize_velocity(v_out, omega_out)



def main():
    base_url = "http://localhost:8000"
    receiver = BotReceiver(base_url)
    start_gps, start_orientation = receiver.moving_average(10)
    destination_gps = (38.828540, -77.306340)
    
    nav_system = RobotNavigationSystem(base_url, start_gps, destination_gps)
    nav_system.run()

if __name__ == "__main__":
    main()