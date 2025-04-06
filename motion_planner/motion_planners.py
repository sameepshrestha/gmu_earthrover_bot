from motion_planner.dwa import DWA
from motion_planner.configs import MotionPlannerType, DWAConfiguration


def get_motion_planner(planner_type):
    if planner_type == MotionPlannerType.dwa:
        return DWA(v_range=DWAConfiguration.v_range,
                   a_range=DWAConfiguration.a_range,
                   a_max=DWAConfiguration.a_max,
                   time_step=DWAConfiguration.time_step,
                   predict_time=DWAConfiguration.predict_time,
                   to_goal_cost_gain=DWAConfiguration.to_goal_cost_gain,
                   speed_cost_gain=DWAConfiguration.speed_cost_gain,
                   obs_cost_gain=DWAConfiguration.obs_cost_gain,
                   radius=DWAConfiguration.radius,
                   goal_threshold=DWAConfiguration.goal_threshold,
                   v_resolution=DWAConfiguration.v_resolution,
                   w_resolution=DWAConfiguration.w_resolution,
                   lidar_fov=DWAConfiguration.lidar_fov,
                   lidar_size=DWAConfiguration.lidar_size)
    else:
        raise ValueError("the motion planner type {} is not defined".format(planner_type))
