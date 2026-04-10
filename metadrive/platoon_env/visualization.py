import csv
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


class PlatoonEpisodeRecorder:
    def __init__(self, config):
        self.config = config
        self.viz_config = config["visualization"]
        self.output_root = Path(self.viz_config["output_dir"])
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.episode_dir = None
        self.records = []
        self.summary = {}
        self.frame_index = 0

    def reset(self, env):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = self.output_root / f"episode_{stamp}_seed_{env.current_seed}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        if self.viz_config["save_frames"]:
            (self.episode_dir / "frames").mkdir(parents=True, exist_ok=True)
        if self.viz_config["save_plot"]:
            (self.episode_dir / "plots").mkdir(parents=True, exist_ok=True)
        self.records = []
        self.summary = {}
        self.frame_index = 0

    def append(self, step_data):
        self.records.append(step_data)

    def save_frame(self, image):
        if not self.viz_config["save_frames"] or self.episode_dir is None:
            return
        frame_path = self.episode_dir / "frames" / f"frame_{self.frame_index:05d}.npy"
        try:
            import numpy as np
            np.save(frame_path, image)
            self.frame_index += 1
        except Exception:
            pass

    def save_schematic_frame(self, image: Image.Image):
        if not self.viz_config["save_frames"] or self.episode_dir is None:
            return
        frame_path = self.episode_dir / "frames" / f"frame_{self.frame_index:05d}.png"
        image.save(frame_path)
        self.frame_index += 1

    def finalize(self, env, terminated=None, truncated=None):
        if self.episode_dir is None:
            return
        self.summary = dict(
            total_steps=len(self.records),
            scenario_complete=env.runtime_metrics.get("scenario_complete", False),
            cut_in_complete=env.runtime_metrics.get("cut_in_complete", False),
            cut_in_hold_steps=env.runtime_metrics.get("cut_in_hold_steps", 0),
            terminated=terminated or {},
            truncated=truncated or {},
        )
        with open(self.episode_dir / "episode_records.json", "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary, "steps": self.records}, f, indent=2)
        self._write_vehicle_csv()
        self._write_gap_csv()
        self._write_plots_if_possible()

    def _write_vehicle_csv(self):
        if not self.records:
            return
        vehicle_path = self.episode_dir / "vehicle_timeseries.csv"
        fieldnames = [
            "step", "agent_id", "x", "y", "speed_km_h", "lane_index", "target_speed_km_h", "steering",
            "throttle_brake", "gap_error", "actual_gap", "cut_in_triggered", "cut_in_complete"
        ]
        with open(vehicle_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                for agent_id, info in record["vehicles"].items():
                    writer.writerow(
                        dict(
                            step=record["step"],
                            agent_id=agent_id,
                            x=info["position"][0],
                            y=info["position"][1],
                            speed_km_h=info["speed_km_h"],
                            lane_index=info["lane_index"],
                            target_speed_km_h=info.get("target_speed_km_h"),
                            steering=info["action"][0],
                            throttle_brake=info["action"][1],
                            gap_error=info.get("gap_error"),
                            actual_gap=info.get("actual_gap"),
                            cut_in_triggered=record["cut_in_triggered"],
                            cut_in_complete=record["cut_in_complete"],
                        )
                    )

    def _write_gap_csv(self):
        if not self.records:
            return
        gap_path = self.episode_dir / "gap_timeseries.csv"
        fieldnames = ["step", "leader_follower_1", "follower_1_follower_2", "cut_in_to_follower_1", "follower_2_to_cut_in"]
        with open(gap_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                row = dict(step=record["step"])
                row.update(record["gaps"])
                writer.writerow(row)

    def _write_plots_if_possible(self):
        if not self.viz_config["save_plot"] or not self.records:
            return
        plt = _try_import_matplotlib()
        if plt is None:
            with open(self.episode_dir / "plots" / "README.txt", "w", encoding="utf-8") as f:
                f.write("matplotlib is unavailable in the current environment, so CSV/JSON logs were exported instead.\n")
            return

        steps = [r["step"] for r in self.records]
        vehicle_ids = list(self.records[0]["vehicles"].keys())

        fig, ax = plt.subplots(figsize=(10, 5))
        for agent_id in vehicle_ids:
            ax.plot(steps, [r["vehicles"][agent_id]["speed_km_h"] for r in self.records], label=agent_id)
        ax.set_title("Speed vs Time")
        ax.set_xlabel("step")
        ax.set_ylabel("speed_km_h")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.episode_dir / "plots" / "speed_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for agent_id in vehicle_ids:
            ax.plot(steps, [r["vehicles"][agent_id]["action"][1] for r in self.records], label=agent_id)
        ax.set_title("Throttle/Brake vs Time")
        ax.set_xlabel("step")
        ax.set_ylabel("throttle_brake")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.episode_dir / "plots" / "longitudinal_control_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        for key in self.records[0]["gaps"].keys():
            ax.plot(steps, [r["gaps"][key] for r in self.records], label=key)
        ax.set_title("Gap vs Time")
        ax.set_xlabel("step")
        ax.set_ylabel("meters")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.episode_dir / "plots" / "gap_curve.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, [r["vehicles"]["cut_in_vehicle"]["lateral"] for r in self.records], label="cut_in_lateral")
        ax.plot(steps, [r["vehicles"]["cut_in_vehicle"]["lane_index"] for r in self.records], label="cut_in_lane_index")
        ax.set_title("Cut-In Lateral State")
        ax.set_xlabel("step")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.episode_dir / "plots" / "cut_in_curve.png")
        plt.close(fig)


class PlatoonVisualizer:
    VEHICLE_COLORS = {
        "leader": (220, 60, 60),
        "follower_1": (60, 120, 220),
        "follower_2": (60, 170, 110),
        "cut_in_vehicle": (240, 170, 40),
    }

    def __init__(self, env):
        self.env = env
        self.viz_config = env.config["visualization"]
        self.recorder = PlatoonEpisodeRecorder(env.config)
        self._font = ImageFont.load_default()

    def reset(self):
        self.recorder.reset(self.env)

    def render_step(self, step_data):
        text = None
        if self.viz_config["enable_debug_text"]:
            text = self._build_text(step_data)
        image = None
        if self.viz_config["enable_render"] or self.viz_config["save_video"] or self.viz_config["save_frames"]:
            if self.viz_config["enable_render"] or self.viz_config["save_video"]:
                image = self.env.render(
                    mode="top_down",
                    text=text,
                    film_size=tuple(self.viz_config["film_size"]),
                    screen_size=tuple(self.viz_config["screen_size"]),
                    camera_position=tuple(self.viz_config["camera_position"]) if self.viz_config["camera_position"] else None,
                    num_stack=self.viz_config["num_stack"] if self.viz_config["enable_traj_vis"] else 1,
                    history_smooth=self.viz_config["history_smooth"] if self.viz_config["enable_traj_vis"] else 0,
                    show_agent_name=True,
                    window=self.viz_config["enable_render"],
                    screen_record=self.viz_config["save_video"],
                )
        if image is not None:
            self.recorder.save_frame(image)
        elif self.viz_config["save_frames"]:
            self.recorder.save_schematic_frame(self._draw_schematic_frame(step_data))

        if self.viz_config["enable_terminal_log"] and step_data["step"] % self.viz_config["log_every"] == 0:
            print(self._build_terminal_line(step_data))

    def finalize(self, terminated=None, truncated=None):
        self.recorder.finalize(self.env, terminated=terminated, truncated=truncated)
        if self.viz_config["save_video"] and self.env.top_down_renderer is not None:
            try:
                self.env.top_down_renderer.generate_gif(
                    str(self.recorder.episode_dir / "episode.gif"),
                    duration=max(len(self.recorder.records), 1)
                )
            except Exception:
                pass

    def _build_text(self, step_data):
        text = {
            "step": step_data["step"],
            "cut_in_triggered": step_data["cut_in_triggered"],
            "cut_in_complete": step_data["cut_in_complete"],
            "leader_gap": round(step_data["gaps"]["leader_follower_1"], 2),
            "f1_gap": round(step_data["gaps"]["follower_1_follower_2"], 2),
        }
        for agent_id, info in step_data["vehicles"].items():
            text[f"{agent_id}_v"] = round(info["speed_km_h"], 2)
            text[f"{agent_id}_u"] = tuple(round(v, 3) for v in info["action"])
        return text

    def _build_terminal_line(self, step_data):
        pieces = [
            f"step={step_data['step']}",
            f"cut_in_triggered={step_data['cut_in_triggered']}",
            f"cut_in_complete={step_data['cut_in_complete']}",
        ]
        for agent_id, info in step_data["vehicles"].items():
            pieces.append(
                "{}:pos=({:.2f},{:.2f}) v={:.2f} lane={} u=({:.3f},{:.3f})".format(
                    agent_id,
                    info["position"][0],
                    info["position"][1],
                    info["speed_km_h"],
                    info["lane_index"],
                    info["action"][0],
                    info["action"][1],
                )
            )
        pieces.append(
            "gaps=({:.2f},{:.2f},{:.2f},{:.2f})".format(
                step_data["gaps"]["leader_follower_1"],
                step_data["gaps"]["follower_1_follower_2"],
                step_data["gaps"]["cut_in_to_follower_1"],
                step_data["gaps"]["follower_2_to_cut_in"],
            )
        )
        return " | ".join(pieces)

    def _draw_schematic_frame(self, step_data):
        width, height = tuple(self.viz_config["screen_size"])
        image = Image.new("RGB", (width, height), (245, 247, 250))
        draw = ImageDraw.Draw(image)

        road_margin_x = 70
        road_top = 160
        road_bottom = height - 210
        road_left = road_margin_x
        road_right = width - road_margin_x
        draw.rounded_rectangle((road_left, road_top, road_right, road_bottom), radius=18, fill=(75, 78, 84))

        lane_centers = self._lane_centers(road_top, road_bottom)
        mid_y = (lane_centers[0] + lane_centers[1]) / 2
        self._draw_dashed_line(draw, (road_left + 20, mid_y), (road_right - 20, mid_y), fill=(240, 240, 240), width=3)

        min_x, max_x = self._get_scene_x_range(step_data)
        x_span = max(max_x - min_x, 60.0)
        min_y, max_y = self._get_scene_y_range(step_data)
        y_span = max(max_y - min_y, 1.0)

        if self.viz_config["enable_traj_vis"]:
            self._draw_history(draw, step_data, road_left, road_right, road_top, road_bottom, min_x, x_span, min_y, y_span)

        for agent_id, info in step_data["vehicles"].items():
            px = self._world_x_to_pixel(info["position"][0], road_left, road_right, min_x, x_span)
            py = self._world_y_to_pixel(info["position"][1], road_top, road_bottom, min_y, y_span)
            color = self.VEHICLE_COLORS.get(agent_id, (120, 120, 120))
            self._draw_vehicle(draw, px, py, color, agent_id, info)

        self._draw_hud(draw, step_data, width, height)
        return image

    def _get_scene_x_range(self, step_data):
        x_values = {agent_id: v["position"][0] for agent_id, v in step_data["vehicles"].items()}
        if self.viz_config.get("focus_on_cut_in", False):
            front = max(x_values["leader"], x_values["follower_1"], x_values["cut_in_vehicle"])
            back = min(x_values["follower_2"], x_values["cut_in_vehicle"])
            min_x = back - self.viz_config.get("padding_back", 25.0)
            max_x = front + self.viz_config.get("padding_front", 35.0)
        else:
            min_x = min(x_values.values()) - 12.0
            max_x = max(x_values.values()) + 30.0
        return min_x, max_x

    def _get_scene_y_range(self, step_data):
        y_values = [v["position"][1] for v in step_data["vehicles"].values()]
        min_y = min(y_values) - 4.0
        max_y = max(y_values) + 4.0
        return min_y, max_y

    def _draw_history(self, draw, step_data, road_left, road_right, road_top, road_bottom, min_x, x_span, min_y, y_span):
        history = self.recorder.records[-self.viz_config["num_stack"]:]
        for agent_id in step_data["vehicles"].keys():
            color = self.VEHICLE_COLORS.get(agent_id, (120, 120, 120))
            points = []
            for record in history:
                info = record["vehicles"][agent_id]
                px = self._world_x_to_pixel(info["position"][0], road_left, road_right, min_x, x_span)
                py = self._world_y_to_pixel(info["position"][1], road_top, road_bottom, min_y, y_span)
                points.append((px, py))
            if len(points) >= 2:
                draw.line(points, fill=color, width=2)

    @staticmethod
    def _lane_centers(road_top, road_bottom):
        lane_height = (road_bottom - road_top) / 2.0
        return [road_top + lane_height * 0.5, road_top + lane_height * 1.5]

    @staticmethod
    def _draw_dashed_line(draw, start, end, fill, width=2, dash=14, gap=10):
        x0, y0 = start
        x1, y1 = end
        total = x1 - x0
        cursor = 0
        while cursor < total:
            draw.line((x0 + cursor, y0, min(x0 + cursor + dash, x1), y1), fill=fill, width=width)
            cursor += dash + gap

    @staticmethod
    def _world_x_to_pixel(x, road_left, road_right, min_x, x_span):
        return int(road_left + (x - min_x) / x_span * (road_right - road_left))

    @staticmethod
    def _world_y_to_pixel(y, road_top, road_bottom, min_y, y_span):
        ratio = (y - min_y) / y_span
        return int(road_top + ratio * (road_bottom - road_top))

    def _draw_vehicle(self, draw, px, py, color, agent_id, info):
        half_len = 28
        half_wid = 12
        draw.rounded_rectangle((px - half_len, py - half_wid, px + half_len, py + half_wid), radius=6, fill=color)
        draw.text((px - half_len, py - 34), agent_id, fill=(20, 20, 20), font=self._font)
        draw.text(
            (px - half_len, py + 18),
            "v={:.1f} u={:.2f}".format(info["speed_km_h"], info["action"][1]),
            fill=(20, 20, 20),
            font=self._font,
        )

    def _draw_hud(self, draw, step_data, width, height):
        hud_top = height - 175
        draw.rounded_rectangle((28, hud_top, width - 28, height - 28), radius=16, fill=(255, 255, 255))
        lines = [
            "step={}  cut_in_triggered={}  cut_in_complete={}".format(
                step_data["step"], step_data["cut_in_triggered"], step_data["cut_in_complete"]
            ),
            "gap leader-f1={:.2f}  f1-f2={:.2f}  cutin-f1={:.2f}  f2-cutin={:.2f}".format(
                step_data["gaps"]["leader_follower_1"],
                step_data["gaps"]["follower_1_follower_2"],
                step_data["gaps"]["cut_in_to_follower_1"],
                step_data["gaps"]["follower_2_to_cut_in"],
            ),
        ]
        for agent_id, info in step_data["vehicles"].items():
            lines.append(
                "{} lane={} target_lane={} v={:.2f} target_v={:.2f} steer={:.3f} throttle={:.3f} gap_err={:.2f}".format(
                    agent_id,
                    info["lane_index"],
                    info["target_lane_index"],
                    info["speed_km_h"],
                    info["target_speed_km_h"],
                    info["action"][0],
                    info["action"][1],
                    info["gap_error"],
                )
            )
        for idx, line in enumerate(lines):
            draw.text((42, hud_top + 14 + idx * 20), line, fill=(20, 20, 20), font=self._font)
