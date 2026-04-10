from typing import Dict, Optional

import numpy as np

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.constants import TerminationState
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.platoon_env.config import (
    CUT_IN_VEHICLE, FOLLOWER_1, FOLLOWER_2, LEADER, PLATOON_AGENT_IDS, PLATOON_ENV_DEFAULT_CONFIG
)
from metadrive.platoon_env.controllers import (
    PIDCutInController, PIDFollowerController, RuleBasedLeaderController
)
from metadrive.platoon_env.observation import PlatoonStateObservation
from metadrive.platoon_env.visualization import PlatoonVisualizer
from metadrive.utils import Config
from metadrive.utils.math import clip


class FourVehiclePlatoonEnv(MultiAgentMetaDrive):
    @classmethod
    def default_config(cls) -> Config:
        config = MultiAgentMetaDrive.default_config()
        config.update(PLATOON_ENV_DEFAULT_CONFIG, allow_add_new_key=True)
        return config

    def __init__(self, config=None):
        self.role_order = PLATOON_AGENT_IDS
        self.cut_in_front_target = FOLLOWER_1
        self.cut_in_back_target = FOLLOWER_2
        self.controllers = {}
        self.controller_debug = {}
        self.runtime_metrics = {}
        self.visualizer = None
        super().__init__(config=config)
        self.visualizer = PlatoonVisualizer(self)

    def setup_engine(self):
        # This fixed four-vehicle scenario does not use MARL respawn, so we intentionally
        # skip registering SpawnManager to avoid the built-in "agent0/agent1/..." assumption.
        MetaDriveEnv.setup_engine(self)

    def _post_process_config(self, config):
        config = MetaDriveEnv._post_process_config(self, config)
        config["num_agents"] = len(self.role_order)
        config["is_multi_agent"] = True
        config["allow_respawn"] = False

        platoon_cfg = config["platoon"]
        spawn_road = tuple(platoon_cfg["spawn_road"])
        spacing = float(platoon_cfg["inter_vehicle_spacing"])
        base_long = float(platoon_cfg["initial_longitudinal"])
        initial_speed = platoon_cfg["initial_speed_km_h"] / 3.6
        platoon_lane_index = int(platoon_cfg["platoon_lane_index"])
        cut_in_lane_index = int(platoon_cfg["cut_in_lane_index"])

        spawn_spec = {
            LEADER: dict(spawn_lane_index=(*spawn_road, platoon_lane_index), spawn_longitude=base_long + 2 * spacing),
            FOLLOWER_1: dict(spawn_lane_index=(*spawn_road, platoon_lane_index), spawn_longitude=base_long + spacing),
            FOLLOWER_2: dict(spawn_lane_index=(*spawn_road, platoon_lane_index), spawn_longitude=base_long),
            CUT_IN_VEHICLE: dict(spawn_lane_index=(*spawn_road, cut_in_lane_index), spawn_longitude=base_long),
        }

        agent_configs = {}
        for agent_id in self.role_order:
            v_cfg = config["vehicle_config"].copy(unchangeable=False)
            v_cfg.update(
                dict(
                    use_special_color=(agent_id == LEADER),
                    random_color=(agent_id != LEADER),
                    destination=None,
                    spawn_velocity=[initial_speed, 0.0],
                    spawn_velocity_car_frame=True,
                )
            )
            v_cfg.update(spawn_spec[agent_id])
            agent_configs[agent_id] = v_cfg
        config["agent_configs"] = agent_configs
        return config

    def get_single_observation(self):
        return PlatoonStateObservation(self.config)

    def reset(self, seed: Optional[int] = None):
        if self.visualizer is not None and self.visualizer.recorder.records:
            self.visualizer.finalize()
        obses, infos = super().reset(seed=seed)
        self._build_controllers()
        self._update_runtime_metrics(reset=True)
        if self.visualizer is not None:
            self.visualizer.reset()
            self._record_current_step()
        return obses, infos

    def _get_reset_return(self, reset_info):
        self._update_runtime_metrics(reset=True)
        return super()._get_reset_return(reset_info)

    def step(self, actions=None):
        if actions is None:
            actions = {}
        merged_actions = self._merge_actions(actions)
        merged_actions = self._preprocess_actions(merged_actions)
        engine_info = self._step_simulator(merged_actions)
        while self.in_stop:
            self.engine.taskMgr.step()
        self._update_runtime_metrics(reset=False)
        obs, rewards, terminated, truncated, info = self._get_step_return(merged_actions, engine_info=engine_info)
        terminated["__all__"] = all(terminated.values())
        truncated["__all__"] = all(truncated.values())
        result = (obs, rewards, terminated, truncated, info)
        self._record_current_step(terminated=result[2], truncated=result[3])
        return result

    def _build_controllers(self):
        platoon_cfg = self.config["platoon"]
        self.controllers = {
            LEADER: RuleBasedLeaderController(platoon_cfg["leader_controller"]),
            FOLLOWER_1: PIDFollowerController(platoon_cfg["follower_controller"], LEADER),
            FOLLOWER_2: PIDFollowerController(platoon_cfg["follower_controller"], FOLLOWER_1),
            CUT_IN_VEHICLE: PIDCutInController(platoon_cfg["cut_in_controller"]),
        }
        for agent_id, controller in self.controllers.items():
            controller.reset(self, agent_id)

    def _merge_actions(self, external_actions: Dict[str, np.ndarray]):
        merged = {}
        external_agents = set(self.config["platoon"]["external_control_agents"])
        self.controller_debug = {}
        for agent_id in self.role_order:
            if agent_id == LEADER:
                merged[agent_id] = self.controllers[agent_id].act(self, agent_id)
                self.controller_debug[agent_id] = dict(self.controllers[agent_id].last_debug)
                continue
            if agent_id in external_agents and agent_id in external_actions:
                merged[agent_id] = np.asarray(external_actions[agent_id], dtype=np.float32)
                self.controller_debug[agent_id] = dict(
                    controller="external",
                    steering=float(merged[agent_id][0]),
                    throttle_brake=float(merged[agent_id][1]),
                )
            else:
                merged[agent_id] = self.controllers[agent_id].act(self, agent_id)
                self.controller_debug[agent_id] = dict(self.controllers[agent_id].last_debug)
        return merged

    def get_target_lane(self, vehicle, target_lane_index: int):
        ref_lanes = vehicle.navigation.current_ref_lanes if vehicle.navigation is not None else None
        if ref_lanes is None or len(ref_lanes) == 0:
            return vehicle.lane
        target_lane_index = int(clip(target_lane_index, 0, len(ref_lanes) - 1))
        return ref_lanes[target_lane_index]

    def get_longitudinal_gap(self, rear_vehicle, front_vehicle) -> float:
        return float(front_vehicle.position[0] - rear_vehicle.position[0])

    def get_reference_front_vehicle(self, agent_id: str):
        if agent_id == LEADER:
            return None
        if agent_id == FOLLOWER_1:
            return self.agents[LEADER]
        if agent_id == FOLLOWER_2:
            return self.agents[FOLLOWER_1]
        if agent_id == CUT_IN_VEHICLE:
            return self.agents[self.cut_in_front_target]
        return None

    def _update_runtime_metrics(self, reset=False):
        if reset or not self.agents:
            hold_steps = 0
        else:
            hold_steps = self.runtime_metrics.get("cut_in_hold_steps", 0)

        cut_in_vehicle = self.agents.get(CUT_IN_VEHICLE, None)
        cut_in_complete = False
        cut_in_triggered = False
        if cut_in_vehicle is not None:
            controller = self.controllers.get(CUT_IN_VEHICLE, None)
            cut_in_triggered = bool(controller.triggered) if controller is not None else False
            _, lat = cut_in_vehicle.lane.local_coordinates(cut_in_vehicle.position)
            cut_in_complete = (
                cut_in_vehicle.lane.index[-1] == self.config["platoon"]["platoon_lane_index"] and
                abs(lat) <= self.config["platoon"]["completion_lateral_threshold"]
            )
        if cut_in_complete:
            hold_steps += 1
        else:
            hold_steps = 0
        completion_hold = max(self.config["platoon"]["completion_hold_steps"], 1)
        self.runtime_metrics = dict(
            cut_in_triggered=cut_in_triggered,
            cut_in_complete=cut_in_complete,
            cut_in_hold_steps=hold_steps,
            cut_in_hold_progress=min(hold_steps / completion_hold, 1.0),
            scenario_complete=hold_steps >= completion_hold,
        )

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        any_crash = any(
            v.crash_vehicle or v.crash_object or v.crash_building or v.crash_human or v.crash_sidewalk
            for v in self.agents.values()
        )
        any_out_of_road = any(self._is_out_of_road(v) for v in self.agents.values())
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        scenario_complete = self.runtime_metrics.get("scenario_complete", False)
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: scenario_complete,
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            "any_crash": any_crash,
            "any_out_of_road": any_out_of_road,
            "scenario_complete": scenario_complete,
            "cut_in_complete": self.runtime_metrics.get("cut_in_complete", False),
            "cut_in_hold_steps": self.runtime_metrics.get("cut_in_hold_steps", 0),
        }
        done_info[TerminationState.CRASH] = any_crash
        done = any_crash or any_out_of_road or scenario_complete
        if max_step and self.config["truncate_as_terminate"]:
            done = True
        return done, done_info

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        cfg = self.config["platoon"]
        reward_cfg = cfg["reward"]
        target_lane = self.get_target_lane(
            vehicle,
            cfg["platoon_lane_index"] if vehicle_id != CUT_IN_VEHICLE or self.runtime_metrics["cut_in_triggered"] else
            cfg["cut_in_lane_index"]
        )
        _, lane_error = target_lane.local_coordinates(vehicle.position)
        target_speed = cfg["leader_target_speed_km_h"] if vehicle_id == LEADER else (
            cfg["cut_in_target_speed_km_h"] if vehicle_id == CUT_IN_VEHICLE else cfg["follower_target_speed_km_h"]
        )
        reward = 0.0
        reward -= reward_cfg["speed_weight"] * abs(vehicle.speed_km_h - target_speed)
        reward -= reward_cfg["lane_weight"] * abs(lane_error)

        front_vehicle = self.get_reference_front_vehicle(vehicle_id)
        gap_error = 0.0
        if front_vehicle is not None:
            gap_error = abs(self.get_longitudinal_gap(vehicle, front_vehicle) - cfg["desired_gap"])
            reward -= reward_cfg["gap_weight"] * gap_error

        if self._is_out_of_road(vehicle):
            reward = -reward_cfg["out_of_road_penalty"]
        elif vehicle.crash_vehicle or vehicle.crash_object or vehicle.crash_building or vehicle.crash_sidewalk:
            reward = -reward_cfg["crash_penalty"]
        elif vehicle_id == CUT_IN_VEHICLE and self.runtime_metrics["cut_in_complete"]:
            reward += reward_cfg["cut_in_completion_bonus"]

        step_info = dict(
            role=vehicle_id,
            speed_km_h=float(vehicle.speed_km_h),
            target_speed_km_h=float(target_speed),
            lane_index=int(vehicle.lane.index[-1]),
            lane_error=float(lane_error),
            gap_error=float(gap_error),
            cut_in_triggered=self.runtime_metrics["cut_in_triggered"],
            cut_in_complete=self.runtime_metrics["cut_in_complete"],
        )
        return reward, step_info

    def cost_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        cost = 0.0
        if self._is_out_of_road(vehicle):
            cost = 1.0
        elif vehicle.crash_vehicle or vehicle.crash_object or vehicle.crash_building or vehicle.crash_sidewalk:
            cost = 1.0
        return cost, {"cost": cost}

    def _record_current_step(self, terminated=None, truncated=None):
        if self.visualizer is None or not self.agents:
            return
        step_data = self._collect_step_data()
        self.visualizer.recorder.append(step_data)
        self.visualizer.render_step(step_data)
        if terminated is not None and (terminated.get("__all__", False) or truncated.get("__all__", False)):
            self.visualizer.finalize(terminated=terminated, truncated=truncated)

    def _collect_step_data(self):
        vehicles = {}
        for agent_id, vehicle in self.agents.items():
            _, lateral = vehicle.lane.local_coordinates(vehicle.position)
            debug = self.controller_debug.get(agent_id, {})
            vehicles[agent_id] = dict(
                position=[float(vehicle.position[0]), float(vehicle.position[1])],
                speed_km_h=float(vehicle.speed_km_h),
                lane_index=int(vehicle.lane.index[-1]),
                lateral=float(lateral),
                heading_theta=float(vehicle.heading_theta),
                action=[
                    float(debug.get("steering", vehicle.current_action[0] if vehicle.current_action is not None else 0.0)),
                    float(debug.get("throttle_brake", vehicle.current_action[1] if vehicle.current_action is not None else 0.0)),
                ],
                target_speed_km_h=float(debug.get("target_speed_km_h", vehicle.speed_km_h)),
                target_lane_index=int(debug.get("target_lane_index", vehicle.lane.index[-1])),
                gap_error=float(debug.get("gap_error", 0.0)),
                actual_gap=float(debug.get("actual_gap", 0.0)),
                controller_debug=debug,
                crash_vehicle=bool(vehicle.crash_vehicle),
                out_of_road=bool(self._is_out_of_road(vehicle)),
            )
        step_data = dict(
            step=int(self.episode_step),
            cut_in_triggered=bool(self.runtime_metrics.get("cut_in_triggered", False)),
            cut_in_complete=bool(self.runtime_metrics.get("cut_in_complete", False)),
            scenario_complete=bool(self.runtime_metrics.get("scenario_complete", False)),
            gaps=dict(
                leader_follower_1=float(self.get_longitudinal_gap(self.agents[FOLLOWER_1], self.agents[LEADER])),
                follower_1_follower_2=float(self.get_longitudinal_gap(self.agents[FOLLOWER_2], self.agents[FOLLOWER_1])),
                cut_in_to_follower_1=float(self.get_longitudinal_gap(self.agents[CUT_IN_VEHICLE], self.agents[FOLLOWER_1])),
                follower_2_to_cut_in=float(self.get_longitudinal_gap(self.agents[FOLLOWER_2], self.agents[CUT_IN_VEHICLE])),
            ),
            vehicles=vehicles,
        )
        return step_data

    def close(self):
        if self.visualizer is not None and self.visualizer.recorder.records:
            self.visualizer.finalize()
        super().close()
