import numpy as np
import torch
import random
import os
import sys

sys.path.append(r"/home/kangning/kangning/motion-transformer/decision-transformer/gym")

from torch.utils.tensorboard import SummaryWriter
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg


#path = r"/home/wenbin/kangning/motion/RIOT/decision-transformer/gym/motion_test_dataset"
#files = os.listdir(path)
#a = torch.cuda.is_available()

state_dim = 99
max_ep_len = 1000

file_path = '/home/kangning/kangning/motion-transformer/decision-transformer/gym/motion_train_dataset/9d723ab7-ba38-400d-a680-68a2e288db28.npy'
model = torch.load('/home/kangning/kangning/motion-transformer/decision-transformer/gym/dt_model/99dim_eps1000_with_retri_loss_mask_goal_20window_cuowei.pt')

data = np.load(file_path, allow_pickle=True)
od_dict = data.item()

rotation_traj_first_state = od_dict["rotation"]["arr"][0]
root_traj_first_state = od_dict["root_translation"]["arr"][0]

traj_len = len(od_dict["rotation"]["arr"])

rotation_traj_first_state = rotation_traj_first_state.reshape(-1, )

initial_state = np.concatenate((rotation_traj_first_state, root_traj_first_state))

traj = []
rotation_traj = od_dict["rotation"]["arr"]
root_traj = od_dict["root_translation"]["arr"]
rotation_traj = rotation_traj.reshape(traj_len, -1)
for i in range(traj_len):
    traj.append(np.concatenate((rotation_traj[i], root_traj[i]),axis=0))

traj = np.array(traj)
initial_state = np.array(initial_state)


states = evaluate_episode_rtg(
    initial_state=initial_state,
    traj=traj,
    env=None,
    state_dim=state_dim,
    act_dim=None,
    model=model,
    max_ep_len=1000,
)
states = states.detach().cpu().numpy()
save_path = 'infer_state_expert_with_retri_loss_mask_goal_cuowei_traj3.npy'
np.save(save_path, states)

'''
for i in range(10):
    file_path = path + '/' + files[i]
    data = np.load(file_path, allow_pickle=True)
    od_dict = data.item()

    rotation_traj = od_dict["rotation"]["arr"][0]
    root_traj = od_dict["root_translation"]["arr"][0]
    traj_len = len(od_dict["rotation"]["arr"])
    rotation_traj = rotation_traj.reshape(-1, )
    initial_state = np.concatenate((rotation_traj, root_traj))
    initial_state = np.array(initial_state)
    states = evaluate_episode_rtg(
        initial_state=initial_state,
        env=None,
        state_dim=state_dim,
        act_dim=None,
        model=model,
        max_ep_len=1000,
    )
    states = states.detach().cpu().numpy()
    save_path = 'infer_state{}'.format(i)
    np.save(save_path, states)
#rint(states.shape)
'''

'''
state_preds = state_preds.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0]
loss = torch.mean((state_preds - state_target) ** 2)
writer.add_scalar("test_loss", loss, i)
'''