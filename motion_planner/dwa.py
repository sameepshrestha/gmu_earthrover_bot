import time
import numpy as np
import math
from multiprocessing import Queue
from motion_planner.base_policy import BasePolicy
from motion_planner.configs import InputKeys
import json


def calculate_dist(a, b):
    return np.linalg.norm([np.array(a[0:2]) - np.array(b[0:2])])


def motion(x, u, dt):
    # motion model
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[2] += u[1] * dt
    x[3] = u[0]
    x[4] = u[1]
    return x


class DWA(BasePolicy):
    def __init__(self, v_range=(0., 1.), a_range=(-1.0, 1.0), a_max=(3., 5.), time_step=0.1, predict_time=2.0,
                 to_goal_cost_gain=3.0, speed_cost_gain=0.1, obs_cost_gain=0.3, radius=0.5, goal_threshold=0.5,
                 v_resolution=0.3, w_resolution=0.3, lidar_fov=(-2 * math.pi / 3, 2 * math.pi / 3), lidar_size=341):
        super(DWA, self).__init__(goal_threshold=goal_threshold, time_step=time_step)
        self.linear_velocity_range = v_range  # [m/s]
        self.angular_velocity_range = a_range  # [rad/s]
        self.acceleration_max = a_max  # [m/ss], [rad/ss] acceleration threshold
        self.data_to_save = []
        self.v_resolution = v_resolution  # [m/s] resolution to choose linear velocity
        self.w_resolution = w_resolution  # [rad/s] resolution to choose angular velocity
        self.predict_time = predict_time  # [s]

        self.to_goal_cost_gain = to_goal_cost_gain  # lower = detour
        self.speed_cost_gain = speed_cost_gain  # lower = faster
        self.obs_cost_gain = obs_cost_gain  # lower z= fearless

        self.robot_radius = radius  # [m]

        self.lidar_fov = lidar_fov
        self.lidar_size = lidar_size
        self.lidar_angles = (lidar_fov[1] - lidar_fov[0]) / self.lidar_size * np.array(range(self.lidar_size)) + lidar_fov[0]
        self.start_time = time.time()

        self.velocity_num = 0

        self.results = []

    def _reset(self):
        self.velocity_num = 0

    def get_dynamic_window(self, current_v, current_a):
        #  [vmin, vmax, a_min, a_max]
        return [max(self.linear_velocity_range[0], current_v - self.acceleration_max[0] * self.time_step),
                min(self.linear_velocity_range[1], current_v + self.acceleration_max[0] * self.time_step),
                max(self.angular_velocity_range[0], current_a - self.acceleration_max[1] * self.time_step),
                min(self.angular_velocity_range[1], current_a + self.acceleration_max[1] * self.time_step)]

    # Calculate a trajectory sampled across a prediction time
    def calc_trajectory(self, xinit, v, w):
        x = np.array(xinit)
        traj = np.array(x)  # many motion models stored per trajectory
        time = 0
        while time <= self.predict_time:
            # store each motion model along a trajectory
            x = motion(x, [v, w], self.time_step)
            traj = np.vstack((traj, x))
            time += self.time_step  # next sample

        return traj

    # Calculate trajectory, costings, and return velocities to apply to robot
    def _predict(self, x, u, dw, ob, goal):
        xinit = x[:]
        min_cost = 10000.0
        min_u = u
        min_u[0] = 0.0
        for v in np.arange(dw[0], dw[1], self.v_resolution):
            for w in np.arange(dw[2], dw[3], self.w_resolution):
                final_cost = self.calc_single_final_input(xinit, v, w, ob, goal=goal)
                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    min_u = [v, w]
                # print("finish vw:{}".format(time.time() - self.start_time))
                # self.start_time = time.time()
        return min_u

    def calc_single_final_input(self, xinit, v, w, ob, goal, queue: Queue = None):
        traj = self.calc_trajectory(xinit, v, w)
        # calc costs with weighted gains
        to_target_cost = self.calc_to_goal_cost(traj=traj, goal=goal) * self.to_goal_cost_gain
        speed_cost = self.speed_cost_gain * (self.linear_velocity_range[1] - traj[-1, 3])
        ob_cost = self.calc_obstacle_cost(traj, ob) * self.obs_cost_gain
        final_cost = to_target_cost + speed_cost + ob_cost
        if queue is not None:
            queue.put((final_cost, (v, w)))
        return final_cost

    # Calculate obstacle cost inf: collision, 0:free
    def calc_obstacle_cost(self, traj, vertices):
        skip_n = 5
        minr = float("inf")
        # Loop through every obstacle in set and calc Pythagorean distance
        # Use robot radius to determine if collision
        for ii in range(0, len(traj[:, 1]), skip_n):
            for vertex in vertices:
                r = calculate_dist(vertex, traj[ii])
                if r <= self.robot_radius:
                    return float("Inf")  # collision
                if minr >= r:
                    minr = r
        return 1.0 / minr

    # Calculate goal cost via Pythagorean distance to robot
    def calc_to_goal_cost(self, traj, goal):
        return calculate_dist(goal, traj[-1])

    def convert_lidar_to_vertices(self, lidar, pose):
        lidar[np.where(lidar > 10)] = 0
        lidar = lidar
        xs = np.array(lidar) * np.cos(self.lidar_angles)
        ys = np.array(lidar) * np.sin(self.lidar_angles)
        global_xs = math.cos(pose[-1]) * xs - math.sin(pose[-1]) * ys + pose[0]
        global_ys = math.sin(pose[-1]) * xs + math.cos(pose[-1]) * ys + pose[1]
        vertices = np.transpose(np.array([global_xs, global_ys]))
        # plt.figure(1)
        # plt.plot(global_xs, global_ys)
        # plt.show()
        return vertices[np.where(lidar>0)]
        # return vertices

    def step(self, obs):
        """
        goal: x, y in global frame
        pose: x, y, theta in global frame
        velocity: linear, angular
        obstacles: lidar data
        """
        goal = obs[InputKeys.goal]
        pose = obs[InputKeys.pose]
        velocity = obs[InputKeys.velocity]
        lidar = obs[InputKeys.lidar]
        position = [pose[0], pose[1], pose[-1], 0, 0]
        vertices = self.convert_lidar_to_vertices(pose=pose, lidar=lidar)
        dw = self.get_dynamic_window(current_v=velocity[0], current_a=velocity[1])
        action = self._predict(x=position, u=velocity, dw=dw, ob=vertices, goal=goal)
        return action

    # def step(self, obs):
    #     # try:
    #         """
    #         goal: x, y in global frame
    #         pose: x, y, theta in global frame
    #         velocity: linear, angular
    #         obstacles: lidar data
    #         """
    #         goal = obs[InputKeys.goal]
    #         pose = obs[InputKeys.pose]
    #         velocity = obs[InputKeys.velocity]
    #         lidar = obs[InputKeys.lidar]
    #
    #         position = [pose[0], pose[1], pose[-1], 0, 0]
    #         vertices = self.convert_lidar_to_vertices(pose=pose, lidar=lidar)
    #         dw = self.get_dynamic_window(current_v=velocity[0], current_a=velocity[1])
    #         action = self._predict(x=position, u=velocity, dw=dw, ob=vertices, goal=goal)
    #
    #         # inputs = [position[0], position[1], position[2], velocity[0], velocity[1], goal[0], goal[1]]
    #         # outputs = [action[0], action[1]]
    #         # Store position and velocity in a list
    #         # self.data_to_save.append({
    #         #     "input": inputs,
    #         #     "action": outputs
    #         # })
    #         return action
        # finally:
        #     # Save the data when exiting the method, regardless of the reason
        #     with open('/home/saketh/Desktop/position_action_data.txt', 'w') as file:
        #         for data in self.data_to_save:
        #             file.write(json.dumps(data) + '\n')