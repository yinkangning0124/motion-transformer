import numpy as np
import torch


def evaluate_episode(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    mode="normal",
    state_mean=0.0,
    state_std=1.0,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length

'''
for decision transformer
'''
def evaluate_episode_rtg(
    initial_state,
    traj,
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=1000,
    device="cuda",
    target_return=None,
    mode="normal",
):

    model.eval()
    model.to(device=device)

    goal_state = traj[20]

    initial_state = (
        torch.from_numpy(initial_state)
       .reshape(1, state_dim)
       .to(device=device, dtype=torch.float32)
    )

    goal_state = (
        torch.from_numpy(goal_state)
        .reshape(1, state_dim)
        .to(device=device, dtype=torch.float32)
    )
    
    states = torch.cat((goal_state, initial_state), dim=0)
    


    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long)],
           dim=1,
       )

    for t in range(320):
        if t >= len(traj) -4:
            goal_state = (
                torch.from_numpy(traj[-1])
                .reshape(1, state_dim)
                .to(device=device, dtype=torch.float32)
        )
        else:
            goal_state = (
                torch.from_numpy(traj[t + 3])
                .reshape(1, state_dim)
                .to(device=device, dtype=torch.float32)
        )

        state = model.get_action(
            states.to(dtype=torch.float32),
            goal_state.to(dtype=torch.float32),
            #actions.to(dtype=torch.float32),
            #rewards.to(dtype=torch.float32),
            #target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )

        state = state.detach().cpu().numpy()
        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)

        timesteps = torch.cat(
            [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 2)],
            dim=1,
        )

    return states
