import numpy as np
import torch

import time

from decision_transformer.training.buffer import ReplayBuffer
# from ..training.rollout_pid import *
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(
        self,
        optimizer,
        batch_size,
        loss_fn,
        model=None,
        get_batch=None,
        scheduler=None,
        act_dim=None,
        device=None,
        trajectory=None,
        optimizer1=None,
        optimizer2=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.act_dim = act_dim
        self.device = device
        self.trajectory = trajectory
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_offline_iteration(self, env, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        writer = SummaryWriter("motion_transformer_99dim_eps1000")
        for i in range(num_steps):
            train_loss = self.train_step(
                env
            )  # here are the state and state_pred after one iteration
            writer.add_scalar("train_loss", train_loss, i)
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs["time/training"] = time.time() - train_start

        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        return logs

    def train_online_iteration(self, env, num_steps, iter_num=0, print_logs=False):
        train_losses = []
        logs = dict()
        next_states = []

        replay_buffer = ReplayBuffer(
            buffer_size=500000,
            batch_size=self.batch_size,
            seed=1000,
            device=self.device,
        )

        sample_traj = self.trajectory[0:50]  # 使用0-50条轨迹作为监督数据

        env.reset()
        for traj in sample_traj:
            for i in range(len(traj["observations"]) - 1):
                current_state = traj["observations"][i]
                next_state = traj["observations"][i + 1]
                action = pid_act(
                    env=env,
                    current_obs=current_state,
                    next_obs=next_state,
                )
                qpos = current_state[0:5]
                qpos = np.concatenate(
                    (
                        np.zeros(1),
                        qpos,
                    )
                )
                qvel = current_state[5:11]
                env.set_state(qpos, qvel)
                next_env_state,_ , _, _ = env.step(action)
                next_env_state = np.array(next_env_state)
                error = np.mean((next_env_state - next_state) ** 2)
                reward = np.exp(-error)
                replay_buffer.add_experience(
                    states=current_state,
                    actions=action,
                    next_states=next_state,
                    rewards=reward,
                )

        """
        # provide expertise trajectories action
        sample_traj = self.trajectory[0:50]

        for traj in sample_traj:
            for i in range(len(traj["observations"]) - 1):
                current_state = traj["observations"][i]
                current_action = traj["actions"][i]
                next_state = traj["observations"][i + 1]
                replay_buffer.add_experience(
                    states=current_state,
                    actions=current_action,
                    next_states=next_state,
                    rewards=np.zeros_like(current_state),
                )
        """

        train_start = time.time()

        # i think it should be better to set replay buffer in this function
        self.model.train()

        writer = SummaryWriter("runs_RL_1iter_with_conditioned_std_clipreward")
        for i in range(num_steps):
            train_loss, rewards, log_prob, action, std = self.train_step(
                env, replay_buffer, iter_num
            )
            action_x = action[:, 0].max()
            action_y = action[:, 1].max()
            action_z = action[:, 2].max()
            writer.add_scalar("log_prob", log_prob, i)
            writer.add_scalar("loss", train_loss, i)
            writer.add_scalar("reward", rewards, i)
            writer.add_scalar("action_x", action_x, i)
            writer.add_scalar("action_y", action_y, i)
            writer.add_scalar("action_z", action_z, i)
            writer.add_scalar("std", std, i)
            train_losses.append(train_loss)
            # if self.scheduler is not None:
            # self.scheduler.step()

        logs["time/training"] = time.time() - train_start

        logs["training/train_loss_mean"] = np.mean(train_losses)
        logs["training/train_loss_std"] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print("=" * 80)
            print(f"Iteration {iter_num}")
            for k, v in logs.items():
                print(f"{k}: {v}")

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(
            self.batch_size
        )
        state_target, action_target, reward_target = (
            torch.clone(states),
            torch.clone(actions),
            torch.clone(rewards),
        )

        state_preds, action_preds, reward_preds = self.model.forward(
            states,
            actions,
            rewards,
            masks=None,
            attention_mask=attention_mask,
            target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds,
            action_preds,
            reward_preds,
            state_target[:, 1:],
            action_target,
            reward_target[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
