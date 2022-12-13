import numpy as np
import torch
from decision_transformer.training.trainer import Trainer


class InverseDynamicsTrainer(Trainer):
    def train_step(self, env, replay_buffer, iter_num):
        states, action_targets, next_states, rewards = replay_buffer.sample()

        
        net_input = torch.cat((states, next_states), 1)
        net_output = self.model(net_input, return_log_prob=True, log_std_fixed=True)
        # action_preds, log_prob1 = net_output[0], net_output[3]
        action_preds, log_prob1, std = net_output[0], net_output[3], net_output[5]
        #action_preds = torch.clamp(action_preds, -1, 1)
        action_for_step = action_preds.detach().cpu().numpy()

        next_states_after_action = []

        action_targets = action_targets.detach().cpu().numpy()
        for i in range(self.batch_size):
            env.reset()
            qpos = states.detach().cpu().numpy()[i, 0:5]
            qpos = np.concatenate(
                (
                    np.zeros(1),
                    qpos,
                )
            )
            qvel = states.detach().cpu().numpy()[i, 5:11]
            env.set_state(qpos, qvel)
            if iter_num <= 5:
                next_state_after_action, _, _, _ = env.step(action_targets[i])
                # the type of the output of env.step() is list
            else:
                next_state_after_action, _, _, _ = env.step(action_for_step[i])
            next_states_after_action.append(next_state_after_action)
        next_states_after_action = np.array(next_states_after_action)
        next_states_after_action = torch.from_numpy(next_states_after_action).cuda(
            device=self.device
        )


        if iter_num > 5:
            for j in range(self.batch_size):
                    error = (next_states_after_action[j] - next_states[j]) ** 2
                    error_mean = error.mean()
                    if error_mean < 0.15:
                        error = torch.exp(-((next_states_after_action[j] - next_states[j]) ** 2))
                        rewards[j] = torch.mean(error)
                    else:
                        reward = - error - 1
                        rewards[j] = reward.mean()
                    # error = torch.exp(-((next_states_after_action[j] - next_states[j]) ** 2))
                    # error = ((next_states[j] - next_states_after_action[j]) ** 2)
                    # rewards[j] = torch.mean(error)
                    # print(rewards[j])


        with torch.no_grad():
            self.diagnostics["state_error"] = (
                torch.mean((next_states_after_action - next_states) ** 2)
                .detach()
                .cpu()
                .item()
            )
        
        action_targets = torch.from_numpy(action_targets).float().to(self.device)
        if iter_num <= 5:
            self.optimizer1.zero_grad()
            loss = self.loss_fn(action_preds, action_targets)
            loss.backward()
            self.optimizer1.step()

        else:
            self.optimizer2.zero_grad()
            loss = -torch.mean(rewards * log_prob1)
            loss.backward()
            self.optimizer2.step()
            

        # nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)

        """
        with torch.no_grad():
            self.diagnostics["action_error"] = (
                torch.mean((action_preds - action_targets) ** 2).detach().cpu().item()
            )

        states = states.detach().cpu().numpy()
        action_preds = action_preds.detach().cpu().numpy()
        next_states = next_states.detach().cpu().numpy()
        rewards = rewards.detach().cpu().numpy()

        for k in range(self.batch_size):
            replay_buffer.add_experience(
                states=states[k],
                actions=action_preds[k],
                next_states=next_states[k],
                rewards=rewards[k],
            )
        """

        return (
            loss.detach().cpu().item(),
            rewards.mean().detach().cpu().numpy(),
            log_prob1.mean().detach().cpu().numpy(),
            action_preds.detach().cpu().numpy(),
            std.mean().detach().cpu().numpy(),
        )
