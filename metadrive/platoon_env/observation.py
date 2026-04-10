import gymnasium as gym
import numpy as np

from metadrive.obs.observation_base import BaseObservation
from metadrive.utils.math import clip


class PlatoonStateObservation(BaseObservation):
    OTHER_AGENT_DIM = 4
    EGO_DIM = 6
    FRONT_DIM = 2
    SCENARIO_DIM = 2

    def __init__(self, config):
        super().__init__(config)
        self.env = None
        self._agent_order = []

    @property
    def observation_space(self):
        dim = self.EGO_DIM + self.FRONT_DIM + self.SCENARIO_DIM + 3 * self.OTHER_AGENT_DIM
        return gym.spaces.Box(-1.0, 1.0, shape=(dim, ), dtype=np.float32)

    def reset(self, env, vehicle=None):
        self.env = env
        self._agent_order = list(env.role_order)

    def observe(self, vehicle):
        env = self.env
        cfg = self.config["platoon"]
        agent_id = env.agent_manager.object_to_agent(vehicle.name)
        max_speed = cfg["max_speed_km_h"]
        observe_distance = cfg["observation_distance"]
        lane_count = max(env.current_map.config["lane_num"], 1)
        long, lat = vehicle.lane.local_coordinates(vehicle.position)
        ego = [
            clip(long / observe_distance, -1.0, 1.0),
            clip(lat / vehicle.lane.width, -1.0, 1.0),
            clip(vehicle.speed_km_h / max_speed, -1.0, 1.0),
            float(np.sin(vehicle.heading_theta)),
            float(np.cos(vehicle.heading_theta)),
            clip(vehicle.lane.index[-1] / max(lane_count - 1, 1), 0.0, 1.0),
        ]

        front_vehicle = env.get_reference_front_vehicle(agent_id)
        if front_vehicle is None:
            front_features = [1.0, 0.0]
        else:
            gap = env.get_longitudinal_gap(vehicle, front_vehicle)
            front_features = [
                clip(gap / observe_distance, -1.0, 1.0),
                clip((front_vehicle.speed_km_h - vehicle.speed_km_h) / max_speed, -1.0, 1.0),
            ]

        scenario = [
            1.0 if env.runtime_metrics["cut_in_triggered"] else 0.0,
            clip(env.runtime_metrics["cut_in_hold_progress"], 0.0, 1.0),
        ]

        others = []
        for other_id in self._agent_order:
            if other_id == agent_id:
                continue
            other_vehicle = env.agents[other_id]
            rel = vehicle.convert_to_local_coordinates(other_vehicle.position, vehicle.position)
            rel_speed = other_vehicle.speed_km_h - vehicle.speed_km_h
            others.extend(
                [
                    clip(rel[0] / observe_distance, -1.0, 1.0),
                    clip(rel[1] / observe_distance, -1.0, 1.0),
                    clip(rel_speed / max_speed, -1.0, 1.0),
                    clip((other_vehicle.lane.index[-1] - vehicle.lane.index[-1]) / max(lane_count - 1, 1), -1.0, 1.0),
                ]
            )

        obs = np.asarray(ego + front_features + scenario + others, dtype=np.float32)
        return obs
