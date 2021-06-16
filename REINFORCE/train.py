import gym
import numpy as np
import torch


def train(env, agent, num_episodes, learning_rate, gamma=0.5):

    def discount_rewards(reward, gamma):
        reward = np.array(reward)
        discount = np.array([gamma**i for i in range(len(reward))])
        discounted = np.array([reward[i:].dot(discount[i:]) for i in range(len(reward))])
        return discounted - discounted.mean()

    action_trajectory = []
    states_trajectory = []
    rewards_trajectory = []
    total = []
    
    optimizer = torch.optim.Adam(agent.parameters(), lr = learning_rate)
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        action_hist = []
        state_hist = []
        reward_hist = []

        while not done:
            action = np.random.choice(np.arange(env.action_space.n), p = agent.forward(obs).detach().numpy())
            state_hist.append(obs)
            action_hist.append(action)
            obs, reward, done, _ = env.step(action)
            reward_hist.append(reward)
        
        unadjusted_reward = sum(reward_hist)
        reward_hist = discount_rewards(reward_hist, gamma = 0.99)

        action_trajectory.extend(action_hist)
        states_trajectory.extend(state_hist)
        rewards_trajectory.extend(reward_hist)

        total.append(unadjusted_reward)

    state_tensor = torch.FloatTensor(states_trajectory)
    action_tensor = torch.LongTensor(action_trajectory)
    reward_tensor = torch.FloatTensor(rewards_trajectory)

    assert len(state_tensor) == len(action_tensor) == len(reward_tensor)

    optimizer.zero_grad()

    p_log = torch.log(agent.forward(state_tensor))
    prob_w_reward = reward_tensor * torch.gather(p_log, dim = 1, index = action_tensor.unsqueeze(1)).squeeze()
    
    loss = -prob_w_reward.mean()
    loss.backward()
    optimizer.step()

    return sum(total)/len(total)