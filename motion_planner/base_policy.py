from abc import ABC, abstractmethod


class BasePolicy(ABC):
    def __init__(self, goal_threshold=0.5, time_step=0.1):
        self.step_number = 0
        self.goal_threshold = goal_threshold  # [m]
        self.time_step = time_step  # [s]

    @abstractmethod
    def _reset(self):
        """"""

    def reset(self):
        self.step_number = 0
        self._reset()

    def predict(self, observation):
        action = self.step(obs=observation)
        self.step_number += 1
        return action

    @abstractmethod
    def step(self, obs):
        """

        :param obs:
        :return:
        """

