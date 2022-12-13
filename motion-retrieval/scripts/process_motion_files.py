from typing import List, Dict, Tuple
import os
import hydra

# issacgym is required to be imported before torch
import isaacgym  # noqa
import torch
import numpy as np
from omegaconf import DictConfig

from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.tasks.humanoid_ampmy import build_amp_observations


def fetch_all_demo_trajs(env) -> List[np.ndarray]:
    dt = env.dt
    motion_lib = env._motion_lib

    motion_steps = np.floor(motion_lib._motion_lengths / dt).astype(int)
    motion_files = motion_lib._motion_files
    motion_times, motion_ids = [], []
    for idx in range(len(motion_steps)):
        motion_times.append(np.arange(motion_steps[idx]) * dt)
        motion_ids.append(
            np.ones(motion_steps[idx], dtype=int) * motion_lib.motion_ids[idx].item()
        )
    motion_times = np.concatenate(motion_times)
    motion_ids = np.concatenate(motion_ids)

    (
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        key_pos,
    ) = env._motion_lib.get_motion_state(motion_ids, motion_times)
    root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
    raw_obs = torch.cat((root_states[:, 3:7], dof_pos), dim=-1)

    amp_obs_demo = build_amp_observations(
        root_states, dof_pos, dof_vel, key_pos, env._local_root_obs
    )

    amp_demo_raw_obs, amp_demo_trajs, amp_demo_infos = [], [], []
    for idx in range(len(motion_steps)):
        amp_demo_raw_obs.append(
            raw_obs[idx * motion_steps[idx] : (idx + 1) * motion_steps[idx]]
        )
        amp_demo_trajs.append(
            amp_obs_demo[idx * motion_steps[idx] : (idx + 1) * motion_steps[idx]]
        )
        amp_demo_infos.append(
            [(motion_files[idx], i * dt) for i in np.arange(motion_steps[idx])]
        )
    return amp_demo_raw_obs, amp_demo_trajs, amp_demo_infos


def raw_trajs_to_dataset(
    demo_raw_obs, demo_trajs: List[np.ndarray], demo_infos: List[Tuple[str, int]], future_steps: int = 5
) -> Dict[str, np.ndarray]:
    raw_observations, raw_future_observations, observations, future_observations, infos = [], [], [], [], []
    traj_start_idxes, traj_end_idxes = [], []
    for raw_obs_traj, obs_traj, info in zip(demo_raw_obs, demo_trajs, demo_infos):
        traj_start_idxes.append(len(raw_observations))
        for idx in range(len(obs_traj) - future_steps):
            raw_observations.append(raw_obs_traj[idx])
            raw_future_observations.append(raw_obs_traj[idx + 1 : idx + future_steps + 1])
            observations.append(obs_traj[idx])
            future_observations.append(obs_traj[idx + 1 : idx + future_steps + 1])
            infos.append(info[idx])
        traj_end_idxes.append(len(raw_observations))

    raw_observations = np.stack(raw_observations, axis=0)
    raw_future_observations = np.stack(raw_future_observations, axis=0)
    observations = np.stack(observations, axis=0)
    future_observations = np.stack(future_observations, axis=0)
    infos = np.stack(infos, axis=0)
    traj_start_idxes = np.stack(traj_start_idxes, axis=0)
    traj_end_idxes = np.stack(traj_end_idxes, axis=0)
    return {
        "raw_observations": raw_observations,
        "raw_future_observations": raw_future_observations,
        "observations": observations,
        "future_observations": future_observations,
        "infos": infos,
        "traj_start_idxes": traj_start_idxes,
        "traj_end_idxes": traj_end_idxes,
    }


def train_test_split(all_data: Dict[str, np.ndarray], test_ratio: float = 0.2):
    data_size = len(all_data["observations"])
    train_data_size = int(data_size * (1 - test_ratio))
    train_data, test_data = {k: [] for k in all_data.keys()}, {k: [] for k in all_data.keys()}

    num_trajs = len(all_data["traj_start_idxes"])
    shuf_idx = np.random.permutation(num_trajs)
    data = train_data
    for start_idx, end_idx in zip(all_data["traj_start_idxes"][shuf_idx], all_data["traj_end_idxes"][shuf_idx]):
        data["traj_start_idxes"].append(len(data["observations"]))
        for k, v in all_data.items():
            if k in ["traj_start_idxes", "traj_end_idxes"]:
                continue
            data[k].extend(v[start_idx: end_idx])
        data["traj_end_idxes"].append(len(data["observations"]))
        if len(data["observations"]) >= train_data_size:
            data = test_data

    for k, v in train_data.items():
        train_data[k] = np.stack(v, axis=0)
    for k, v in test_data.items():
        test_data[k] = np.stack(v, axis=0)
    print("\nTraining data size: {}, Test data size: {}\n".format(len(train_data["observations"]), len(test_data["observations"])))

    return train_data, test_data


@hydra.main(config_name="config", config_path="../assets/isaacgymenv_cfg")
def run(cfg: DictConfig) -> None:

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    env = isaacgym_task_map[cfg.task_name](
        cfg=cfg_dict["task"],
        rl_device=cfg.rl_device,
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
        virtual_screen_capture=cfg.capture_video,
        force_render=cfg.force_render,
    )

    process_cfg = cfg_dict["process"]
    np.random.seed(process_cfg["seed"])
    amp_raw_obs, amp_demo_trajs, all_demo_infos = fetch_all_demo_trajs(env)
    all_data = raw_trajs_to_dataset(
        amp_raw_obs, amp_demo_trajs, all_demo_infos, future_steps=process_cfg["future_steps"]
    )
    train_data, test_data = train_test_split(
        all_data, test_ratio=process_cfg["test_ratio"]
    )

    processed_data_path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)),
        ),
        "assets/processed_data",
    )
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    torch.save(
        train_data,
        os.path.join(
            processed_data_path,
            f"train_data-fs_{process_cfg['future_steps']}.th",
        ),
    )
    torch.save(
        test_data,
        os.path.join(
            processed_data_path,
            f"test_data-fs_{process_cfg['future_steps']}.th",
        ),
    )


if __name__ == "__main__":
    run()
