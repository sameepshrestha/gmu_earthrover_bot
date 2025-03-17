from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import math

# --- 1. Semantic Class Definitions ---
label2name_all = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle"
}


def pixel_to_metric(px, py, res,origin,rwidth = 0.00267890625, rheight = 0.003048):
    resx = rwidth 
    resy = rheight
    x_metric = (px - origin[1]) * resx  # Center x around origin

    y_metric =  (origin[0] - py) *resy # Invert y to make forward positive
    return x_metric, y_metric

def metric_to_pixel(x_metric, y_metric, res,origin, rwidth=0.00267890625, rheight=0.003048):
    py = origin[0] - (y_metric / rheight)  # Direct scaling for y
    px = (x_metric / rwidth) + origin[1]  # Direct scaling for x
    return int(px), int(py)

# --- 3. DWA Planner Class Definition ---
class DWAPlanner:
    def __init__(self):
        # 3.1. Robot Configuration
        self.max_speed = 0.833  # [m/s]
        self.min_speed = -0.2   # [m/s]
        self.max_yaw_rate = 1.0  # [rad/s]
        self.max_accel = 2.0     # [m/s^2]
        self.max_delta_yaw_rate = 1.0  # [rad/s^2]
        self.v_resolution = 0.1      # [m/s]
        self.yaw_rate_resolution = 0.05  # [rad/s]
        self.dt = .4                 # [s]
        self.predict_time = 4.0        # [s]
        self.to_goal_cost_gain = 0.3   # Weight for heading cost
        self.speed_cost_gain = 0.15    # Weight for speed cost
        self.obstacle_cost_gain = 0.5  # Weight for semantic/obstacle cost
        self.smoothness_cost_gain = 0.05  # Weight for smoothness cost
        self.target_speed = 0.4       # [m/s]
        self.gaussian_sigma = 0.5
        self.min_yaw_rate = 0.1

                # 3.2. Semantic Cost Map Configuration
        self.COST_MAP = {
            "road": 0,
            "sidewalk": 0.1,
            "building": 1.0,
            "wall": 1.0,
            "fence": 1,
            "pole": 1,
            "traffic light": 1,
            "traffic sign": 1,
            "vegetation": 0.7,
            "terrain": 0.4,
            "sky": 1.0,
            "person": 1.0,
            "rider": 1.0,
            "car": .95,
            "truck": .90,
            "bus": 1.0,
            "train": 1.0,
            "motorcycle": 1.0,
            "bicycle": 1.0
        }

    def process_semantic_map(self, semantic_map, class_indices):
        height, width = semantic_map.shape
        cost_map = np.zeros((height, width))
        for idx, class_name in class_indices.items():
            mask = semantic_map == idx
            cost_map[mask] = self.COST_MAP.get(class_name, 1.0)
        cost_map = gaussian_filter(cost_map, sigma=self.gaussian_sigma)
        print("Cost Map Stats:")
        print(f"  Min Cost: {np.min(cost_map):.3f}, Max Cost: {np.max(cost_map):.3f}, Mean Cost: {np.mean(cost_map):.3f}")
        return cost_map

    def calc_semantic_cost(self, traj_x: List[float], traj_y: List[float],
                           cost_map: np.ndarray, resolution: float, origin: Tuple[int, int]) -> float:
        cost = 0.0
        traversable_threshold = 0.5
        for x, y in zip(traj_x, traj_y):
            px, py = metric_to_pixel(x, y, resolution, origin)
            map_x, map_y = int(px), int(py)
            if 0 <= map_x < cost_map.shape[1] and 0 <= map_y < cost_map.shape[0]:
                cost += cost_map[map_y, map_x]
                if cost_map[map_y, map_x] > traversable_threshold:
                    return float('inf')
            else:
                return float('inf')
        return cost / len(traj_x) if traj_x else 0.0

    def calc_heading_cost(self, traj_x: List[float], traj_y: List[float], traj_yaw: List[float], goal: List[float]) -> float:
        dx = goal[0] - traj_x[-1]
        dy = goal[1] - traj_y[-1]
        angle_to_goal = np.arctan2(dy, dx)
        cost_angle = angle_to_goal - traj_yaw[-1]
        cost = abs(np.arctan2(np.sin(cost_angle), np.cos(cost_angle)))
        return cost

    def calc_speed_cost(self, v: float) -> float:
        return abs(self.target_speed - v)

    def calc_smoothness_cost(self, traj_yaw: List[float]) -> float:
        yaw_diffs = np.diff(traj_yaw)
        return np.sum(np.abs(yaw_diffs))

    def calc_dynamic_window(self, x_current):
        vs_range_v = [self.min_speed, self.max_speed]
        vs_range_omega = [-self.max_yaw_rate, self.max_yaw_rate]
        vd_range_v = [x_current[3] - self.max_accel * self.dt, x_current[3] + self.max_accel * self.dt]
        vd_range_omega = [x_current[4] - self.max_delta_yaw_rate * self.dt, x_current[4] + self.max_delta_yaw_rate * self.dt]
        dw = [max(vs_range_v[0], vd_range_v[0]), min(vs_range_v[1], vd_range_v[1]),
              max(vs_range_omega[0], vd_range_omega[0]), min(vs_range_omega[1], vd_range_omega[1])]
        return dw

    def calc_trajectory(self, x_init: List[float], v: float, omega: float) -> Tuple[List[float], List[float], List[float]]:
        x_traj, y_traj, yaw_traj = [x_init[0]], [x_init[1]], [x_init[2]]
        x_current = list(x_init)
        time = 0.0
        while time <= self.predict_time:
            x_current = self.motion_model(x_current, [v, omega], self.dt)
            x_traj.append(x_current[0])
            y_traj.append(x_current[1])
            yaw_traj.append(x_current[2])
            time += self.dt
        return x_traj, y_traj, yaw_traj

    def motion_model(self, x: List[float], control: List[float], dt: float) -> List[float]:
        v, omega = control
        x_next = list(x)
        x_next[0] += v * np.cos(x[2]) * dt
        x_next[1] += v * np.sin(x[2]) * dt
        x_next[2] += omega * dt
        x_next[3] = v
        x_next[4] = omega
        return x_next

    def calc_control_and_trajectory(self, x_current, goal, cost_map, resolution, origin):
        dw = self.calc_dynamic_window(x_current)
        min_total_cost = float('inf')
        best_control = [0.0, 0.0]
        best_trajectory = None

        velocity_range = np.arange(dw[0], dw[1] + self.v_resolution, self.v_resolution)
        yaw_rate_range = np.arange(dw[2], dw[3] + self.yaw_rate_resolution, self.yaw_rate_resolution)
        for v in velocity_range:
            for omega in yaw_rate_range:
                traj_x, traj_y, traj_yaw = self.calc_trajectory(x_current, v, omega)
                semantic_cost = self.calc_semantic_cost(traj_x, traj_y, cost_map, resolution, origin)
                if semantic_cost == float('inf'):
                    continue
                heading_cost = self.calc_heading_cost(traj_x, traj_y, traj_yaw, goal)
                speed_cost = self.calc_speed_cost(v)
                smoothness_cost = self.calc_smoothness_cost(traj_yaw)
                total_cost = (self.obstacle_cost_gain * semantic_cost +
                              self.to_goal_cost_gain * heading_cost +
                              self.speed_cost_gain * speed_cost +
                              self.smoothness_cost_gain * smoothness_cost)
                # print(semantic_cost,heading_cost,speed_cost,smoothness_cost,total_cost)
                if total_cost < min_total_cost:
                    min_total_cost = total_cost
                    best_control = [v, omega]
                    best_trajectory = [traj_x, traj_y, traj_yaw]
        if best_trajectory is not None:
            return best_control[0], best_control[1], best_trajectory
        else:
            return None, None, None

    def plan(self, x_start, goal, semantic_map, class_indices, resolution, origin):
        cost_map = self.process_semantic_map(semantic_map, class_indices)
        best_v, best_omega, best_trajectory = self.calc_control_and_trajectory(x_start, goal, cost_map, resolution, origin)
        if best_trajectory:
            # Convert trajectory to pixel coordinates for visualization
            traj_px = [metric_to_pixel(x, y, resolution, origin) for x, y in zip(best_trajectory[0], best_trajectory[1])]
            best_trajectory_px = list(zip(*traj_px))  # [(px1, px2, ...), (py1, py2, ...)]
            return best_v, best_omega, best_trajectory, best_trajectory_px, cost_map
        else:
            return None, None, None, None, None

# def local_goal_selection(segmentation_mask, direction="STRAIGHT", row_search=5):
#     h, w = segmentation_mask.shape
#     robot_width = 0.25
#     width_resolution = 0.015
#     robot_width_px = int(robot_width / width_resolution)
#     quantile_map = {"STRAIGHT": 0.5, "LEFT": 0.25, "RIGHT": 0.75}

#     q_value = quantile_map.get(direction, 0.5)
#     for row in range(row_search, h-1, 10):
#         # Identify free indices (road or sidewalk)
#         free_indices = np.where((segmentation_mask[row, :] == 0) | (segmentation_mask[row, :] == 1))[0]
#         if len(free_indices) < robot_width_px:
#             continue
#         chosen_pixel = int(np.quantile(free_indices, q_value))
#         left_limit = chosen_pixel - robot_width_px // 2
#         right_limit = chosen_pixel + robot_width_px // 2
#         print("Chosen pixel:", chosen_pixel, "Left limit:", left_limit, "Right limit:", right_limit)
#         # Check that the region is free (either road or sidewalk)
#         if left_limit >= 0 and right_limit < w and np.all((segmentation_mask[row, left_limit:right_limit] == 0) | (segmentation_mask[row, left_limit:right_limit] == 1)):
#             return chosen_pixel, row  
#     return None, None
def local_goal_selection(segmentation_mask, direction="STRAIGHT", row_search=5):
    h, w = segmentation_mask.shape
    robot_width = 0.25
    width_resolution = 0.015
    robot_width_px = int(robot_width / width_resolution)  # ≈16 pixels
    quantile_map = {"STRAIGHT": 0.5, "LEFT": 0.25, "RIGHT": 0.75}

    q_value = quantile_map.get(direction, 0.5)

    for row in range(row_search, h-1, 10):  # Sparse row search
        # Get the row data
        row_data = segmentation_mask[row, :]
        
        # Find free indices (0 or 1)
        free_mask = (row_data == 0) | (row_data == 1)
        
        # Identify continuous segments
        segments = []
        start = None
        for i in range(w):
            if free_mask[i]:
                if start is None:
                    start = i
            elif start is not None:
                segments.append((start, i - 1))
                start = None
        if start is not None:  # Close last segment if it ends at w-1
            segments.append((start, w - 1))
        
        if not segments:  # No free segments
            continue
        
        # Find the longest segment
        longest_segment = max(segments, key=lambda x: x[1] - x[0] + 1)
        segment_length = longest_segment[1] - longest_segment[0] + 1
        
        if segment_length < robot_width_px:  # Segment too small for robot
            continue
        
        # Get indices of the longest segment
        free_indices = np.arange(longest_segment[0], longest_segment[1] + 1)
        
        # Calculate the chosen pixel based on quantile within this segment
        chosen_pixel = int(np.quantile(free_indices, q_value))
        left_limit = chosen_pixel - robot_width_px // 2
        right_limit = chosen_pixel + robot_width_px // 2
        
        # print(f"Row {row}: Longest segment from {longest_segment[0]} to {longest_segment[1]}, "
        #       f"Length: {segment_length}, Chosen pixel: {chosen_pixel}, "
        #       f"Left limit: {left_limit}, Right limit: {right_limit}")
        
        # Ensure the region fits within bounds and is entirely free
        if (left_limit >= 0 and right_limit < w and
            np.all((row_data[left_limit:right_limit] == 0) | (row_data[left_limit:right_limit] == 1))):
            return chosen_pixel, row
    
    return None, None