import numpy as np 
import os 
from PIL import Image
import time 
import cv2
import matplotlib.pyplot as plt 
import osmnx as ox
from collections import deque
def normalize_velocity(linear_velocity, angular_velocity, 
                       max_linear_speed=0.9722, max_angular_speed=1):  # 3.5 km/hr in m/s (~0.9722 m/s)

    norm_linear = np.clip(linear_velocity / max_linear_speed, 0, 1)
    norm_angular = np.clip(angular_velocity / max_angular_speed, -1, 1)
    return norm_linear, norm_angular

def compute_velocity_from_rpms(rpms, wheel_radius=0.05, wheel_base=0.25):

    rpm_fl, rpm_fr, rpm_rl, rpm_rr = rpms

    def rpm_to_speed(rpm):
        return (rpm * 2 * np.pi * wheel_radius) / 60
    v_left = (rpm_to_speed(rpm_fl) + rpm_to_speed(rpm_rl)) / 2
    v_right = (rpm_to_speed(rpm_fr) + rpm_to_speed(rpm_rr)) / 2
    linear_velocity = (v_left + v_right) / 2
    angular_velocity = (v_right - v_left) / wheel_base
    return linear_velocity, angular_velocity

CITYSCAPES_COLORMAP = np.array([
    [128, 64, 128],  # road
    [255, 255, 200],  # sidewalk
    [70, 70, 70],    # building
    [102, 102, 156], # wall
    [190, 153, 153], # fence
    [153, 153, 153], # pole
    [250, 170, 30],  # traffic light
    [220, 220, 0],   # traffic sign
    [107, 142, 35],  # vegetation
    [152, 251, 152], # terrain
    [70, 130, 180],  # sky
    [220, 20, 60],   # person
    [255, 0, 0],     # rider
    [0, 0, 142],     # car
    [0, 0, 70],      # truck
    [0, 60, 100],    # bus
    [0, 80, 100],    # train
    [0, 0, 230],     # motorcycle
    [119, 11, 32]    # bicycle
], dtype=np.uint8)


def save_data(data, data_type, main_folder="/home/kintou/Work/Robotixx/Sameep_PC/Output"):

    if data_type not in ["image", "segmentation", "cost_map"]:
        raise ValueError("data_type must be 'image', 'segmentation_mask', or 'cost_map'.")

    subfolder = os.path.join(main_folder, data_type + "s")
    os.makedirs(subfolder, exist_ok=True)

    timestamp = int(time.time() * 1000)  # Milliseconds for uniqueness
    filename = f"{data_type}_{timestamp}.png"
    filepath = os.path.join(subfolder, filename)

    if data_type == "image":
        # if not isinstance(data, Image.Image):
        #     raise ValueError("For 'image', data must be a PIL Image.")
        data.save(filepath)
    elif data_type == "segmentation":
        if not (isinstance(data, np.ndarray) and len(data.shape) == 2 and data.dtype == np.uint8):
            raise ValueError("For 'segmentation_mask', data must be a 2D uint8 NumPy array.")
        color_segmentation = CITYSCAPES_COLORMAP[data]
        cv2.imwrite(filepath, cv2.cvtColor(color_segmentation, cv2.COLOR_RGB2BGR))  # OpenCV saves in BGR format

    elif data_type == "cost_map":
        if not (isinstance(data, np.ndarray) and len(data.shape) == 2 and data.dtype in [np.float32, np.float64]):
            raise ValueError("For 'cost_map', data must be a 2D float NumPy array.")
        data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
        data_uint8 = data_normalized.astype(np.uint8)
        colored_cost_map = cv2.applyColorMap(data_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(filepath, colored_cost_map)
    # print(f"Saved {data_type} to {filepath}")



def plot_map(G, current_utm, prev_coords, next_coords, projected_point, route_coords, save_path="/home/kintou/Work/Robotixx/Sameep_PC/Output/gps_maps"):
    frame = int(time.time() * 1000)  # Milliseconds for uniqueness
    fig1, ax1 = ox.plot_graph(G, node_size=0, show=False, close=False)
    # Plot route
    route_x, route_y = zip(*route_coords)
    ax1.plot(route_x, route_y, c='yellow', linewidth=2, label="Route")
    # Plot points in UTM
    ax1.scatter(prev_coords[0], prev_coords[1], c='red', marker='o', s=50, label="Previous Node")
    ax1.scatter(next_coords[0], next_coords[1], c='red', marker='o', s=50, label="Next Node")
    ax1.scatter(current_utm.x, current_utm.y, c='green', marker='x', s=100, label="Current GPS")
    ax1.scatter(projected_point.x, projected_point.y, c='blue', marker='s', s=50, label="Projected Point")
    ax1.legend()
    ax1.set_title(f"Frame {frame:03d} - Tracking Progress")
    plt.savefig(f"{save_path}/frame_{frame:03d}.png")
    plt.close(fig1)


# def visualize_trajectory(cost_map, trajectory_px, start_px, goal_px, save_path="/home/kintou/Work/Robotixx/Sameep_PC/Output/traj_maps"):
#     frame = int(time.time() * 1000) 

#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(cost_map, cmap='viridis', origin='upper')
#     if trajectory_px:
#         traj_x_px, traj_y_px = trajectory_px
#         ax.plot(traj_x_px, traj_y_px, '-r', label='Trajectory')
#     ax.plot(start_px[1], start_px[0], 'bs', markersize=8, label='Start')  # Note: (py, px)
#     ax.plot(goal_px[1], goal_px[0], 'g*', markersize=10, label='Goal')    # Note: (py, px)
#     ax.xlabel("Pixel X")
#     ax.ylabel("Pixel Y")
#     ax.title("Cost Map and DWA Trajectory")
#     ax.legend()
#     ax.colorbar(label='Cost')
#     plt.savefig(f"{save_path}/gps_maps_{frame:03d}.png")
#     plt.close()




def save_transformed_image(transformed_image, save_path = "/home/kintou/Work/Robotixx/Sameep_PC/Output/tranformed_images", filename="transformed_image.png"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    full_path = os.path.join(save_path, filename)
    
    if transformed_image.dtype != np.uint8:
        if transformed_image.max() <= 1.0:
            transformed_image = (transformed_image * 255).astype(np.uint8)
        else:
            transformed_image = transformed_image.astype(np.uint8)
    
    if len(transformed_image.shape) == 2:  # Grayscale (H, W)
        img = Image.fromarray(transformed_image, mode="L")
    elif len(transformed_image.shape) == 3 and transformed_image.shape[2] == 3:  # RGB (H, W, 3)
        img = Image.fromarray(transformed_image, mode="RGB")
    else:
        raise ValueError("Unsupported image shape. Expected 2D (grayscale) or 3D (RGB: H, W, 3).")
    img.save(full_path)

class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def update(self, new_value):
        self.values.append(new_value)

    def get_average(self):
        return sum(self.values) / len(self.values) if self.values else 0.0
def visualize_trajectory_with_opencv(cost_map, trajectory_px, start_px, goal_px, save_path="/home/kintou/Work/Robotixx/Sameep_PC/Output/traj_maps"):
    if cost_map.dtype in (np.float64, np.float32):
        # Shift and scale to use full range (0 becomes 0, 1 becomes 255)
        cost_map_normalized = 255 * (cost_map - cost_map.min()) / (cost_map.max() - cost_map.min() + 1e-6)  # Avoid div by zero
        cost_map_uint8 = cost_map_normalized.astype(np.uint8)
    else:
        cost_map_uint8 = cost_map
    cost_map_with_trajectory = cv2.cvtColor(cost_map_uint8, cv2.COLOR_GRAY2BGR)
    if trajectory_px:
        # print(trajectory_px, start_px, goal_px)
        traj_x_px, traj_y_px = trajectory_px
        if len(traj_x_px) > 1:
            for i in range(len(traj_x_px) - 1):
                cv2.line(cost_map_with_trajectory, 
                         (traj_x_px[i], traj_y_px[i]), 
                         (traj_x_px[i + 1], traj_y_px[i + 1]), 
                         (0, 0, 255), 2)  
                # print((traj_y_px[i], traj_x_px[i]))
    if start_px:
        cv2.circle(cost_map_with_trajectory, (start_px[1], start_px[0]), 8, (255, 0, 0), -1)  
    if goal_px:
        cv2.circle(cost_map_with_trajectory, (goal_px[1], goal_px[0]), 10, (0, 255, 0), -1)    

    frame = int(time.time() * 1000)  # Create a timestamp for unique filenames
    save_file = f"{save_path}/trajectory_map_{frame}.png"
    cv2.imwrite(save_file, cost_map_with_trajectory)
    # Return the processed image (RGB format)
    return cost_map_with_trajectory
class ImcrementatlVisualizer:
    def __init__(self, height= 576, width = 1024, gps_tracker = None):
        self.height = height 
        self. width = width 
        self .img_slot = np.zeros((height, width, 3), dtype= np.uint8)
        self.seg_slot = np.zeros((height, width, 3), dtype= np.uint8)
        self.traj_plot = np.zeros((height, width, 3), dtype = np.uint8)
        self.gps_plot = np.zeros((height, width, 3), dtype = np.uint8)
        self.gps_tracker = gps_tracker
        self.window_name = "Robot_Navigation"
    def update(self, image = None, seg_mask = None, traj_data = None, gps_data = None):
        if image is not None:
            self.img_slot = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self.img_slot = cv2.resize(self.img_slot, (self.width, self.height))

        if seg_mask is not None:
            color_segmentation = CITYSCAPES_COLORMAP[seg_mask]
            seg_vis = cv2.cvtColor(color_segmentation, cv2.COLOR_RGB2BGR)
            self.seg_slot = cv2.resize(seg_vis, (self.width, self.height))
        
        if traj_data is not None:
            self.traj_plot = cv2.resize(traj_data, (self.width, self.height))
            # print(self.traj_plot)
        if gps_data is not None:
            gps_map  = self.gps_tracker.plot_map(gps_data) 
            self.gps_plot = cv2.resize(gps_map,  (self.width, self.height))
        top_row = np.hstack((self.img_slot, self.seg_slot))

        bottom_row = np.hstack((self.traj_plot, self.gps_plot))
        full_vis = np.vstack((top_row, bottom_row))

        cv2.imshow(self.window_name, full_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt("User stopped the visualization")

    def close(self):
        cv2.destroyWindow(self.window_name)


            
