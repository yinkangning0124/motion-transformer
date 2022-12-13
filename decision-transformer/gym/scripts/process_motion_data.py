import numpy as np
import os
import pickle


path = r"/home/wenbin/kangning/motion-transformer/decision-transformer/gym/motion_train_dataset"

files = os.listdir(path=path)

observation_dataset = []
next_observation_dataset = []

for file in files:
    traj = []
    next_traj = []
    filepath = os.path.join(path, file)

    data = np.load(filepath, allow_pickle=True)
    od_dict = data.item()

    rotation_traj = od_dict["rotation"]["arr"]
    root_traj = od_dict["root_translation"]["arr"]

    traj_len = len(root_traj)
    rotation_traj = rotation_traj.reshape(traj_len, -1)

    for i in range(traj_len):
        observation_dataset.append(np.concatenate((root_traj[i], rotation_traj[i]),axis=0))


    for j in range(1, traj_len):
        next_observation_dataset.append(np.concatenate((root_traj[j], rotation_traj[j]),axis=0))
    next_observation_dataset.append(next_observation_dataset[-1])

observation_dataset = np.array(observation_dataset, dtype=object)
next_observation_dataset = np.array(next_observation_dataset, dtype=object)  

#print(observation_dataset.shape)  
#print(next_observation_dataset.shape)

''' check the dataset, minus is zeros means the process is correct
minus = []

for i in range(len(observation_dataset) - 1):
    res = observation_dataset[i + 1] - next_observation_dataset[i]
    minus.append(res)

print(minus)
'''

np.save("observation_dataset.npy", observation_dataset)
np.save("next_observation_dataset.npy", next_observation_dataset)