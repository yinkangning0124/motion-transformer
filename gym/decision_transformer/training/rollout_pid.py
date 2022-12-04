import gym
import numpy as np

import pickle
from scipy.linalg import cho_solve, cho_factor
from mujoco_py import functions as mjf


def compute_desired_accel(env, qpos_err, qvel_err, k_p, k_d):
    dt = env.model.opt.timestep
    nv = env.model.nv

    M = np.zeros(nv * nv)
    mjf.mj_fullM(env.model, M, env.data.qM)
    M.resize(nv, nv)
    # M = M[: env.qvel_lim, : env.qvel_lim]
    # C = env.data.qfrc_bias.copy()[: env.qvel_lim]
    C = env.data.qfrc_bias.copy()

    K_p = np.diag(k_p)
    K_d = np.diag(k_d)
    q_accel = cho_solve(
        cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
        -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
        overwrite_b=True,
        check_finite=False,
    )
    return q_accel.squeeze()


def compute_torque(env, qpos, qvel, target_pos, jkp, jkd):
    # cfg = self.cfg
    # dt = self.model.opt.timestep
    dt = env.model.opt.timestep

    k_p = np.zeros(6)
    k_d = np.zeros(6)
    k_p[3:] = jkp
    k_d[3:] = jkd
    qpos_err = np.concatenate((np.zeros(3), qpos[2:] + qvel[3:] * dt - target_pos))
    qvel_err = qvel
    q_accel = compute_desired_accel(env, qpos_err, qvel_err, k_p, k_d)
    qvel_err += q_accel * dt
    torque = -jkp * qpos_err[3:] - jkd * qvel_err[3:]

    return torque

'''
def save_video(video_frames, filename, fps=24, video_format="mp4"):
    assert fps == int(fps), fps
    import skvideo.io

    skvideo.io.vwrite(
        filename,
        video_frames,
        inputdict={
            "-r": str(int(fps)),
        },
        outputdict={
            "-f": video_format,
            "-pix_fmt": "yuv420p",  # '-pix_fmt=yuv420p' needed for osx https://github.com/scikit-video/scikit-video/issues/74
        },
    )
'''

def pid_act(env, current_obs, next_obs, jkp=5.0, jkd=0.005):
    start_qpos = current_obs[:5]
    start_qvel = current_obs[5:]
    end_qpos = next_obs[:5]

    torque = compute_torque(env, start_qpos, start_qvel, end_qpos[2:], jkp=jkp, jkd=jkd)
    return torque

'''
def experiment(
    name,
    jkp: float = 0.05,
    jkd: float = 0.005,
):
    env = gym.make("Hopper-v3")

    # load dataset
    dataset_path = "data/hopper-expert-v2.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    # for jkp in [5.0, 0.5, 0.05, 0.005, 0.0005, 0.00005]:
    #     for jkd in [5.0, 0.5, 0.05, 0.005, 0.0005, 0.00005]:
    for jkp in [5.0]:
        for jkd in [0.005]:
            for trajectory in trajectories[6:7]:
                control_errors = []
                obs_imgs = []
                current_obs = trajectory["observations"][0]
                for step in range(len(trajectory["observations"]) - 1):
                    end = trajectory["observations"][step + 1]

                    start_qpos = current_obs[:5]
                    start_qvel = current_obs[5:]
                    end_qpos = end[:5]

                    qpos = np.concatenate((np.zeros(1), start_qpos))
                    env.set_state(qpos, start_qvel)
                    torque = compute_torque(env, start_qpos, start_qvel, end_qpos[2:], jkp=jkp, jkd=jkd)
                    current_obs, rew, done, info = env.step(torque.clip(-1, 1))

                    obs_img = env.render(mode="rgb_array")
                    obs_imgs.append(obs_img)

                    control_errors.append(np.linalg.norm(current_obs[2:5] - end[2:5]))
                save_video(obs_imgs, f"test-{jkp}-{jkd}.mp4")
            # search_control_errors.append((jkp, jkd, np.mean(control_errors)))


if __name__ == "__main__":
    experiment("gym-experiment")
'''