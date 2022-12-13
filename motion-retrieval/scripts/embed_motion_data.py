import os
import json
import torch
import joblib
import argparse
import numpy as np

from retrieval.utils import pytorch_util as ptu
from retrieval.utils.dataset import MotionDataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="assets/processed_data/train_data-fs_15.th",
        help="path to the data file to compute embeddings",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        help="path to the model checkpoint file to load",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("-g", "--gpu", help="gpu id", type=int, default=0)
    args = parser.parse_args()
    ptu.set_gpu_mode(True, args.gpu)

    data = torch.load(args.data_file)
    dataloader = MotionDataLoader(
        data=data, batch_size=args.batch_size
    )

    ckpt_data = joblib.load(args.ckpt_file)
    embed_model = ckpt_data["embed_model"].to(ptu.device)

    json_file = os.path.join(os.path.dirname(args.ckpt_file), "variant.json")
    exp_specs = json.load(open(json_file, "r"))

    embeds = []
    for batch in dataloader:
        obs, future_obs = batch
        if exp_specs["condition_on_target"]:
            embed_input = torch.cat([obs, future_obs[:, exp_specs["trainer_params"]["pred_future_steps"] - 1]], dim=1)
        else:
            embed_input = obs
        embed_ = embed_model.embed_network(embed_input.to(ptu.device))
        embeds.append(ptu.get_numpy(embed_))
    embeds = np.concatenate(embeds, axis=0)
    data["embeds"] = embeds

    save_path = os.path.join(
        os.path.dirname(args.ckpt_file),
        args.data_file.split("/")[-1].split(".")[0] + "-embeds.th",
    )
    print("\nSaving processed data to {}\n".format(save_path))
    torch.save(data, save_path)
