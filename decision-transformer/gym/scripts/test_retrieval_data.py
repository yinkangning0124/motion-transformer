import torch
import numpy as np
import sys
import os

retrieval_data_dir = r"/home/kangning/kangning/motion-transformer/motion_data"
retrieval_data_name = "retrieval_data-fs_10_10.th"
data_path = os.path.join(retrieval_data_dir, retrieval_data_name)

retrieval_data = torch.load(data_path)

raw_observations = retrieval_data["raw_observations"]
a = 1
