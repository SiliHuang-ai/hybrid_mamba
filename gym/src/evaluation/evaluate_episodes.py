import numpy as np
import torch
import copy


def evaluate_episode_rtg2(
        env,
        state_dim,
        act_dim,
        z_dim,
        model,
        plan_to_go,
        # plan_encoder,
        horizon,
        K,
        episodes_times,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    context_size = K
    model.eval()
    model.to(device=device)
    model.reset_eval(device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    ep_return = target_return
    sim_states = []
    episode_returns, episode_lengths = [],[]

    for episodes_time in range(episodes_times):
        print("eval_episode:  ",str(episodes_time))
        state = env.reset()
        state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"

        states = state.reshape(1, state_dim)
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)

        t, episode_return, episode_length = 0, 0, 0
        l_states = states[-1].reshape(1, -1)
        l_time_steps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        l_actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        while t < max_ep_len:
            # add padding
            model.step(target_return, None, None, None, None)
            z_distribution_predict = model.step(None, states, None, None, None)

            # rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            tmp_return = target_return[0,-1]
            z_distribution_predict_tmp = z_distribution_predict.repeat([horizon,1])
            h_r = 0
            for i in range(horizon):
                l_actions = torch.cat([l_actions, torch.zeros((1, act_dim), device=device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                action = plan_to_go.get_action(l_states, l_actions, z_distribution_predict_tmp, rewards, l_time_steps)
                l_actions[-1] = action.reshape(-1, act_dim)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
                t+=1
                # cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                cur_state = state.reshape(1, state_dim)
                rewards[-1] = reward
                h_r += reward

                tmp_return = tmp_return - (reward/scale)

                episode_return += reward
                episode_length += 1
                if done:
                    normalized_score = env.get_normalized_score(episode_return)
                    episode_returns.append(normalized_score)
                    episode_lengths.append(episode_length)
                    break

                # if i+1 < horizon:
                l_states = torch.cat([l_states, cur_state], dim=0)
                l_time_steps = torch.cat([l_time_steps, torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)

            # z_actual_distributions = plan_encoder.get_actual_distribution(l_states[:-1],l_actions,l_time_steps[:-1])

            # model.step(None, None, z_distribution_predict, None, None)
            r = torch.tensor(h_r, device=device, dtype=torch.float32).reshape(1, 1)
            model.step(None, None, None, r, None)
            d = torch.tensor(done, device=device, dtype=torch.long).reshape(-1)
            model.step(None, None, None, None, d)

            if done:
                break

            states = cur_state
            target_return = tmp_return.reshape(1, 1)

    # episode_returns, episode_lengths = np.array(episode_returns), np.array(episode_lengths)
    return episode_returns, episode_lengths

def evaluate_episode_rtg3(
        env,
        state_dim,
        act_dim,
        z_dim,
        model,
        plan_to_go,
        plan_encoder,
        horizon,
        K,
        episodes_times,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):
    context_size = K
    model.eval()
    model.to(device=device)
    model.reset_eval(device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    ep_return = target_return
    sim_states = []
    episode_returns, episode_lengths = [],[]

    for episodes_time in range(episodes_times):
        print("eval_episode:  ",str(episodes_time))
        state = env.reset()
        state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
        if mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=state.shape)
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"

        states = state.reshape(1, state_dim)
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)

        t, episode_return, episode_length = 0, 0, 0
        l_states = states[-1].reshape(1, -1)
        l_time_steps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        l_actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        while t < max_ep_len:
            # add padding
            model.step(target_return, None, None, None, None)
            z_distribution_predict = model.step(None, states, None, None, None)

            # rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            tmp_return = target_return[0,-1]
            z_distribution_predict_tmp = z_distribution_predict.repeat([horizon,1])
            h_r = 0
            for i in range(horizon):
                l_actions = torch.cat([l_actions, torch.zeros((1, act_dim), device=device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                action = plan_to_go.get_action(l_states, l_actions, z_distribution_predict_tmp, rewards, l_time_steps)
                l_actions[-1] = action.reshape(-1, act_dim)
                action = action.detach().cpu().numpy()
                state, reward, done, _ = env.step(action)
                state = (torch.from_numpy(state).to(device=device, dtype=torch.float32) - state_mean) / state_std
                t+=1
                # cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
                cur_state = state.reshape(1, state_dim)
                rewards[-1] = reward
                h_r += reward

                tmp_return = tmp_return - (reward/scale)

                episode_return += reward
                episode_length += 1
                if done:
                    normalized_score = env.get_normalized_score(episode_return)
                    episode_returns.append(normalized_score)
                    episode_lengths.append(episode_length)
                    break

                # if i+1 < horizon:
                l_states = torch.cat([l_states, cur_state], dim=0)
                l_time_steps = torch.cat([l_time_steps, torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)

            z_actual_distributions = plan_encoder.get_actual_distribution(l_states[:-1],l_actions,None, l_time_steps[:-1]).unsqueeze(0)

            model.step(None, None, z_actual_distributions, None, None)
            # model.step(None, None, z_distribution_predict, None, None)
            r = torch.tensor(h_r, device=device, dtype=torch.float32).reshape(1, 1)
            model.step(None, None, None, r, None)
            d = torch.tensor(done, device=device, dtype=torch.long).reshape(-1)
            model.step(None, None, None, None, d)

            if done:
                break

            states = cur_state
            target_return = tmp_return.reshape(1, 1)

    # episode_returns, episode_lengths = np.array(episode_returns), np.array(episode_lengths)
    return episode_returns, episode_lengths