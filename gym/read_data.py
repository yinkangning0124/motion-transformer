import numpy as np
import os

path = r"/home/wenbin/kangning/motion/RIOT/decision-transformer/gym/motion_train_dataset"

files = os.listdir(path) # 列出目录下的所有文件
i = 0
for file in files:
    traj = []
    filepath = path + '/' + file
    data = np.load(filepath, allow_pickle=True)
    od_dict = data.item()
    keys = od_dict.keys()
    traj_dofvels = od_dict["dof_vels"]["arr"]
    rotation_traj = od_dict["rotation"]["arr"]
    root_traj = od_dict["root_translation"]["arr"]
    traj_len = len(root_traj)
    rotation_traj = rotation_traj.reshape(traj_len, -1)
    #print(rotation_traj.shape)
    for j in range(traj_len):
        traj.append(np.concatenate((rotation_traj[j], root_traj[j]),axis=0))
    traj = np.array(traj)
    print(traj.shape)

    i = i + 1
    if i == 20:
        break
    
        













'''

a = np.array([1, 2, 3, 4]).reshape(2, 2)
b = np.array([5, 6, 7, 8]).reshape(2, 2)
c = np.concatenate((a, b), axis=1)
print(a)
print(b)
print(c)

'''