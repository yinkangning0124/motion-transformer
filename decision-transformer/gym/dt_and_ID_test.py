'''
integrate decision transformer and inverse dynamics together to
control the motion.
'''
import pickle
import numpy as np
import gym
import argparse
from collections import deque
import torch
import joblib

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)

def test_data(variant):
    env_name = "Hopper-v3"
    env = gym.make(env_name)
    max_len = variant["K"]
    dataset_path = f"data/hopper-expert-v2.pkl"

    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)
    
    dt_model = torch.load('dt_model/1.pt')
    policy = joblib.load('dt_model/best.pkl')["exploration_policy"][0]
    inverse_dynamics = policy.inverse_dynamic
    trajs = trajectories[0 : 50]
    
    env.reset()
    for i in range(len(trajs)):
        test_traj_obs = trajs[i]["observations"]
        test_traj_next_obs = trajs[i]["next_observations"]
        transformer_input = test_traj_obs[0 : 20]
        qpos = test_traj_obs[0, 0:5]
        qpos = np.concatenate(
            (
                np.zeros(1),
                qpos,
            )
        )
        qvel = test_traj_obs[0, 5:11]
        env.set_state(qpos, qvel)
        for j in range(len(test_traj_obs)):
            if j == 0:
                current_state = test_traj_obs[j]
                next_state = test_traj_next_obs[j]
                next_state = torch.from_numpy(next_state).float().to(device="cpu")
            else:
                timesteps = np.arange(0, 20).reshape(1, 20, )
                timesteps = torch.from_numpy(timesteps).float().to(dtype=torch.long, device="cuda")
                attention_mask = np.ones((1, 20, ))
                attention_mask = torch.from_numpy(attention_mask).to(device="cuda")
                Input = transformer_input.reshape(1, max_len, 11)
                Input = torch.from_numpy(Input).float().to(dtype=torch.float32, device="cuda")
                transformer_out = dt_model(Input, timesteps, attention_mask)
                transformer_out = transformer_out.reshape(20, 11)
                next_state = transformer_out[-1] # tensor
                next_state = next_state.to(device="cpu")
            current_state = torch.from_numpy(current_state).float().to(device="cpu")
            
            net_output = inverse_dynamics(current_state, next_state)
            action = net_output[0]
            action = torch.clamp(action, min=-1, max=1)

            action_for_step = action.detach().cpu().numpy()
            real_state, _, done, _ = env.step(action_for_step)
            env.render()
            current_state = real_state
            transformer_input = np.delete(transformer_input, obj=0, axis=0)
            transformer_input = np.insert(transformer_input, obj=-1, values=real_state, axis=0)



        pass
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K",type=int,default=20)

    args = parser.parse_args()


    test_data(variant=vars(args))