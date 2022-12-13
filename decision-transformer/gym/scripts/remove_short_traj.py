import numpy as np
import os

path = r"/home/kangning/kangning/motion-transformer/decision-transformer/gym/motion_train_dataset"
files = os.listdir(path) # 列出目录下的所有文件

remove_list = []

for file in files:
    file_path = os.path.join(path, file)
    data = np.load(file_path, allow_pickle=True)
    od_dict = data.item()
    rotation_traj = od_dict["rotation"]["arr"]
    length = len(rotation_traj)
    if length < 30:
        remove_list.append(file)

print(remove_list)