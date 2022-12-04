import torch
import gym
import pickle
import numpy as np

env = gym.make("Hopper-v3")
dataset_path = f"data/hopper-expert-v2.pkl"

with open(dataset_path, "rb") as f:
    trajectories = pickle.load(f)
#print(trajectories)
env.reset()

Model = torch.load('model_extract/online_model_RL_1iter_with_conditioned_std_clipreward.pt')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Model.to(device)
Model.eval()

for i in range(50):
    traj_test = trajectories[i]["observations"]
    qpos = traj_test[0, 0:5]
    qpos = np.concatenate(
        (
            np.zeros(1),
            qpos,
        )
    )
    qvel = traj_test[0, 5:11]
    env.set_state(qpos, qvel)
    for j in range(len(traj_test) - 1):
        if j == 0:
            current_state = traj_test[j]
        next_state = traj_test[j + 1]
        current_state = torch.from_numpy(current_state).float().to(device="cuda:0")
        next_state = torch.from_numpy(next_state).float().to(device="cuda:0")
        net_input = torch.cat((current_state, next_state))
        net_output = Model.forward(net_input)
        action = net_output[0]
        action = torch.clamp(action, min=-1, max=1)
        print(action)
        action_for_step = action.detach().cpu().numpy()
        real_state, _, _, _ = env.step(action_for_step)
        current_state = real_state
        env.render()

'''

for i in range(50):
    traj_test = trajectories[i]["observations"]
    print(traj_test)
    qpos = traj_test[0, 0:5]
    qpos = np.concatenate(
        (
            np.zeros(1),
            qpos,
        )
    )
    qvel = traj_test[0, 5:11]
    env.set_state(qpos, qvel)
    for j in range(len(traj_test) - 1):
        if j == 0:
            current_state = traj_test[j]
        next_state = traj_test[j + 1]
        current_state = torch.from_numpy(current_state).float().to(device="cuda:0")
        next_state = torch.from_numpy(next_state).float().to(device="cuda:0")
        net_input = torch.cat((current_state, next_state))
        net_output = Model.forward(net_input)
        action = net_output[0]
        action_for_step = action.detach().cpu().numpy()
        real_state, _, _, _ = env.step(action_for_step)
        current_state = real_state
        #error = np.mean((current_state - traj_test[j]) ** 2)
        #print(error)
        env.render()
'''
