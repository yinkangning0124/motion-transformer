import gym
import numpy as np
import torch
import wandb
import os
import argparse
import pickle
import random
import sys

sys.path.append(r"/home/wenbin/kangning/motion-transformer/decision-transformer/gym")

from decision_transformer.evaluation.evaluate_episodes import (
    evaluate_episode,
    evaluate_episode_rtg,
)
from decision_transformer.models.decision_transformer import DecisionTransformer


from decision_transformer.training.seq_trainer import SequenceTrainer

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(
    exp_prefix,
    variant,
):
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    env_name, dataset = variant["env"], variant["dataset"]
    model_type = variant["model_type"]
    group_name = f"{exp_prefix}-{env_name}-{dataset}"
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"

    env = None
    max_ep_len = 1000
    state_dim = 99


    # load dataset
    path = r"/home/wenbin/kangning/motion-transformer/decision-transformer/gym/motion_train_dataset"
    files = os.listdir(path) # 列出目录下的所有文件
    # save all path information into separate lists
    
    print("=" * 50)

    print("=" * 50)

    K = variant["K"]
    batch_size = variant["batch_size"]

    def get_batch(batch_size=256, max_len=20):
        file = random.sample(files, batch_size)

        #s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        s, d, timesteps, mask = [], [], [], []
        for i in range(batch_size):
            traj = []
            filepath = os.path.join(path, file[i])
            data = np.load(filepath, allow_pickle=True)
            od_dict = data.item()
            rotation_traj = od_dict["rotation"]["arr"]
            root_traj = od_dict["root_translation"]["arr"]
            traj_len = len(rotation_traj)
            rotation_traj = rotation_traj.reshape(traj_len, -1)
            for j in range(traj_len):
                traj.append(np.concatenate((rotation_traj[j], root_traj[j]),axis=0))
            
            traj = np.array(traj)
            si = random.randint(0, traj.shape[0] - 1)
            # get sequences from dataset
            s.append(traj[si : si + max_len].reshape(1, -1, state_dim))
            #a.append(traj["actions"][si : si + max_len].reshape(1, -1, act_dim))
            #r.append(traj["rewards"][si : si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            #rtg.append(discount_cumsum(traj["rewards"][si:], gamma=1.0)[: s[-1].shape[1] + 1].reshape(1, -1, 1))
            #if rtg[-1].shape[1] <= s[-1].shape[1]:
            #    rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            #s[-1] = (s[-1] - state_mean) / state_std
            #d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            #rtg[-1] = (np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, d, timesteps, mask

    if model_type == "dt":
        model = DecisionTransformer(
            state_dim=state_dim,
            #act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant["embed_dim"],
            n_layer=variant["n_layer"],
            n_head=variant["n_head"],
            n_inner=4 * variant["embed_dim"],
            activation_function=variant["activation_function"],
            n_positions=1024,
            resid_pdrop=variant["dropout"],
            attn_pdrop=variant["dropout"],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant["learning_rate"],
        weight_decay=variant["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    if model_type == "dt":
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((s_hat - s) ** 2),
        )
    # need to check!
    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant["max_iters"]):
        outputs = trainer.train_offline_iteration(
            env = env, num_steps=variant["num_steps_per_iter"], iter_num=iter + 1, print_logs=True
        )
        if log_to_wandb:
            wandb.log(outputs)
    torch.save(model, 'dt_model/99dim_eps1000.pt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument(
        "--dataset", type=str, default="expert"
    )  # medium, medium-replay, medium-expert, expert
    parser.add_argument(
        "--mode", type=str, default="normal"
    )  # normal for standard setting, delayed for sparse
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--pct_traj", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--model_type", type=str, default="dt"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=10)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", "-w", type=bool, default=False)

    args = parser.parse_args()

    experiment("gym-experiment", variant=vars(args))
