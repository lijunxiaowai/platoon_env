from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import clip, wrap_to_pi


class BaseController(ABC):
    def __init__(self, config):
        self.config = config
        self.last_debug = {}

    def reset(self, env, agent_id: str):
        pass

    @abstractmethod
    def act(self, env, agent_id: str):
        raise NotImplementedError


class PIDLaneTrackingController(BaseController):
    def __init__(self, config):
        super().__init__(config)
        self.speed_pid = PIDController(config["speed_kp"], config["speed_ki"], config["speed_kd"])
        self.heading_pid = PIDController(
            config["steer_heading_kp"], config["steer_heading_ki"], config["steer_heading_kd"]
        )
        self.lateral_pid = PIDController(
            config["steer_lateral_kp"], config["steer_lateral_ki"], config["steer_lateral_kd"]
        )

    def reset(self, env, agent_id: str):
        self.speed_pid.reset()
        self.heading_pid.reset()
        self.lateral_pid.reset()

    def _track_lane(self, vehicle, target_lane) -> float:
        long, lat = target_lane.local_coordinates(vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1.0)
        heading_error = -wrap_to_pi(lane_heading - vehicle.heading_theta)
        steering = self.heading_pid.get_result(heading_error)
        steering += self.lateral_pid.get_result(-lat)
        return float(clip(steering, -1.0, 1.0))

    def _track_speed(self, vehicle, target_speed_km_h: float) -> float:
        speed_error = vehicle.speed_km_h - target_speed_km_h
        throttle = self.speed_pid.get_result(speed_error)
        return float(clip(throttle, -1.0, 1.0))

    def _make_action(self, vehicle, target_lane, target_speed_km_h: float):
        steering = self._track_lane(vehicle, target_lane)
        throttle = self._track_speed(vehicle, target_speed_km_h)
        self.last_debug = dict(
            target_speed_km_h=float(target_speed_km_h),
            target_lane_index=int(target_lane.index[-1]),
            steering=float(steering),
            throttle_brake=float(throttle),
        )
        return np.asarray([steering, throttle], dtype=np.float32)


class RuleBasedLeaderController(PIDLaneTrackingController):
    def act(self, env, agent_id: str):
        vehicle = env.agents[agent_id]
        target_lane = env.get_target_lane(vehicle, env.config["platoon"]["platoon_lane_index"])
        target_speed_km_h = env.config["platoon"]["leader_target_speed_km_h"]
        return self._make_action(vehicle, target_lane, target_speed_km_h)


class PIDFollowerController(PIDLaneTrackingController):
    def __init__(self, config, front_agent_id: str):
        super().__init__(config)
        self.front_agent_id = front_agent_id

    def act(self, env, agent_id: str):
        vehicle = env.agents[agent_id]
        front_vehicle = env.agents[self.front_agent_id]
        platoon_cfg = env.config["platoon"]
        target_lane = env.get_target_lane(vehicle, platoon_cfg["platoon_lane_index"])
        gap = env.get_longitudinal_gap(vehicle, front_vehicle)
        gap_error = gap - platoon_cfg["desired_gap"]
        gap_speed_delta = clip(
            self.config["gap_gain"] * gap_error, -self.config["max_gap_correction_km_h"],
            self.config["max_gap_correction_km_h"]
        )
        target_speed_km_h = clip(
            front_vehicle.speed_km_h + gap_speed_delta, 0.0, platoon_cfg["max_speed_km_h"]
        )
        action = self._make_action(vehicle, target_lane, target_speed_km_h)
        self.last_debug.update(
            dict(
                front_agent_id=self.front_agent_id,
                actual_gap=float(gap),
                desired_gap=float(platoon_cfg["desired_gap"]),
                gap_error=float(gap_error),
                gap_speed_delta=float(gap_speed_delta),
            )
        )
        return action


class PIDCutInController(PIDLaneTrackingController):
    def __init__(self, config):
        super().__init__(config)
        self.triggered = False

    def reset(self, env, agent_id: str):
        super().reset(env, agent_id)
        self.triggered = False

    def _should_trigger(self, env, vehicle) -> bool:
        if self.triggered:
            return True
        trigger_step = self.config["trigger_step"]
        if env.episode_step < trigger_step:
            return False
        longitudinal, _ = vehicle.lane.local_coordinates(vehicle.position)
        if longitudinal < self.config["trigger_min_longitudinal"]:
            return False
        leader_gap = env.get_longitudinal_gap(vehicle, env.agents[env.cut_in_front_target])
        back_gap = env.get_longitudinal_gap(env.agents[env.cut_in_back_target], vehicle)
        return leader_gap >= self.config["min_front_gap"] and back_gap >= self.config["min_back_gap"]

    def act(self, env, agent_id: str):
        vehicle = env.agents[agent_id]
        platoon_cfg = env.config["platoon"]
        self.triggered = self._should_trigger(env, vehicle)
        target_lane_index = platoon_cfg["platoon_lane_index"] if self.triggered else platoon_cfg["cut_in_lane_index"]
        target_lane = env.get_target_lane(vehicle, target_lane_index)
        target_speed_km_h = platoon_cfg["cut_in_target_speed_km_h"]
        action = self._make_action(vehicle, target_lane, target_speed_km_h)
        self.last_debug.update(
            dict(
                triggered=bool(self.triggered),
                current_lane_index=int(vehicle.lane.index[-1]),
                target_lane_index=int(target_lane_index),
                front_gap=float(env.get_longitudinal_gap(vehicle, env.agents[env.cut_in_front_target])),
                back_gap=float(env.get_longitudinal_gap(env.agents[env.cut_in_back_target], vehicle)),
            )
        )
        return action
