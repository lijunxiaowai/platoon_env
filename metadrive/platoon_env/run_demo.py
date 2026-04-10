from metadrive.platoon_env.env import FourVehiclePlatoonEnv


def main():
    env = FourVehiclePlatoonEnv(
        {
            "use_render": False,
            "visualization": {
                "enable_render": False,
                "enable_traj_vis": True,
                "enable_debug_text": True,
                "enable_terminal_log": True,
                "save_video": False,
                "save_frames": True,
                "save_plot": False,
                "focus_on_cut_in": True,
            },
            "platoon": {
                "external_control_agents": [],
            }
        }
    )
    try:
        obs, info = env.reset()
        print("Reset agents:", list(obs.keys()))
        print("Initial cut-in complete:", info["cut_in_vehicle"]["cut_in_complete"])
        for step in range(200):
            obs, reward, terminated, truncated, info = env.step({})
            if step % 20 == 0:
                print(
                    "step={}, cut_in_lane={}, cut_in_complete={}, leader_speed={:.2f}, output_dir={}".format(
                        step,
                        info["cut_in_vehicle"]["lane_index"],
                        info["cut_in_vehicle"]["cut_in_complete"],
                        info["leader"]["speed_km_h"],
                        env.visualizer.recorder.episode_dir,
                    )
                )
            if terminated["__all__"] or truncated["__all__"]:
                print("Episode finished at step", step)
                print("Termination:", terminated)
                print("Truncation:", truncated)
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
