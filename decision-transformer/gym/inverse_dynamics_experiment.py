import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys

from decision_transformer.training.inverse_dynamics_trainer import (
    InverseDynamicsTrainer,
)
from rlkit.torch.common.policies import ReparamMultivariateGaussianPolicy


def experiment(
    exp_prefix,
    variant,
):
    device = variant.get("device", "cuda")
    log_to_wandb = variant.get("log_to_wandb", False)

    env_name, dataset = variant["env"], variant["dataset"]
    group_name = f"{exp_prefix}-{env_name}-{dataset}"
    model_type = variant["model_type"]
    exp_prefix = f"{group_name}-{random.randint(int(1e5), int(1e6) - 1)}"
    batch_size = variant["batch_size"]
    num_hidden = 2
    net_size = 1024

    if env_name == "hopper":
        env = gym.make("Hopper-v3")
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.0  # normalization for rewards/returns
    elif env_name == "halfcheetah":
        env = gym.make("HalfCheetah-v3")
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.0
    elif env_name == "walker2d":
        env = gym.make("Walker2d-v3")
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.0
    elif env_name == "reacher2d":
        from decision_transformer.envs.reacher_2d import Reacher2dEnv

        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.0
    else:
        raise NotImplementedError

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f"data/{env_name}-{dataset}-v2.pkl"
    with open(dataset_path, "rb") as f:
        trajectories = pickle.load(f)

    print("=" * 50)
    print("online training starts!")
    print("=" * 50)

    if model_type == "ID":
        online_model = ReparamMultivariateGaussianPolicy(
            hidden_sizes=num_hidden * [net_size],
            obs_dim=state_dim * 2,
            action_dim=act_dim,
            conditioned_std=False,
        )
    else:
        raise NotImplementedError

    online_model = online_model.to(device=device)

    warmup_steps = variant["warmup_steps"]
    optimizer1 = torch.optim.Adam(
        online_model.parameters(),
        lr=1e-4,
        # weight_decay=variant["weight_decay"],
    )
    optimizer2 = torch.optim.Adam(
        online_model.parameters(),
        #lr=variant["learning_rate"],
        lr=1e-4,
    )
    
    '''
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda steps: min((steps + 1) / warmup_steps, 1)
    )
    '''
    if model_type == "ID":
        online_trainer = InverseDynamicsTrainer(
            model=online_model,
            device=device,
            optimizer1=optimizer1,
            optimizer2=optimizer2,
            batch_size=batch_size,
            loss_fn=lambda a_hat, a: torch.mean((a_hat - a) ** 2),
            act_dim=act_dim,
            trajectory=trajectories,
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project="decision-transformer",
            config=variant,
        )

    for iter in range(variant["max_iters"]):
        outputs = online_trainer.train_online_iteration(
            env=env,
            num_steps=variant["num_steps_per_iter"],
            iter_num=iter + 1,
            print_logs=True,
        )
        if iter == 9:
            torch.save(
                online_model, "model_extract/online_model_RL_1iter_with_conditioned_std_clipreward.pt"
            )


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
        "--model_type", type=str, default="ID"
    )  # dt for decision transformer, bc for behavior cloning
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--n_layer", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--num_eval_episodes", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=20)
    parser.add_argument("--num_steps_per_iter", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_wandb", "-w", type=bool, default=False)

    args = parser.parse_args()

    experiment("gym-experiment", variant=vars(args))
