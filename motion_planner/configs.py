from easydict import EasyDict as edict
import numpy as np

class InputKeys:
    lidar = "lidar"
    velocity = "velocity"
    imu = "imu"
    goal = "goal"
    pose = "pose"
    traversability_map = "traversability_map"
    gps = "gps"


class MotionPlannerStatus:
    Hanging = "hanging"
    Arrived = "arrived"
    Running = "running"


class MotionPlannerType:
    dwa = "dwa"

LidarConfig = edict()
LidarConfig.threshold = 100.0
LidarConfig.frequency = 6.1
LidarConfig.angle_range = 360.0 * np.pi / 180.0
LidarConfig.horizons = 1824

DWAConfiguration = edict()
DWAConfiguration.v_range = (0., 1.)
DWAConfiguration.a_range = (-1.0, 1.0)
DWAConfiguration.a_max = (3., 5.)
DWAConfiguration.time_step = 0.1
DWAConfiguration.predict_time = 2.0
DWAConfiguration.to_goal_cost_gain = 3.0
DWAConfiguration.speed_cost_gain = 0.1
DWAConfiguration.obs_cost_gain = 0.3
DWAConfiguration.radius = 0.5
DWAConfiguration.goal_threshold = 0.5
DWAConfiguration.v_resolution = 0.3
DWAConfiguration.w_resolution = 0.3
DWAConfiguration.lidar_fov = (-LidarConfig.angle_range / 2., LidarConfig.angle_range / 2.) # (-2 * math.pi / 3, 2 * math.pi / 3),    (-math.pi, math.pi)
DWAConfiguration.lidar_size = LidarConfig.scan_horizons


