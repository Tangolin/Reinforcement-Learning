import torch
from torch import nn
import numpy as np
import gym 
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions import Categorical
from torch.optim import Adam, RMSprop
    

class backbone(nn.Module):
    def __init__(self, role, input_neurons, hidden_layer, output_neurons, dropout=0.1):
        super().__init__()
        self.input = input_neurons
        self.hidden = hidden_layer
        self.out = output_neurons
        self.modules = [
            nn.Linear(self.input, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, int(self.hidden/2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden/2) , self.out),
        ]

        if role == 'actor':
            self.modules.append(nn.Softmax(dim=-1))
        
        self.net = nn.Sequential(*self.modules)
    
    def forward(self, inputs):
        inputs = torch.from_numpy(inputs).float()

        return self.net(inputs)


class policy(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
    
    def forward(self, input):
        state_value_function = self.critic(input)
        action_prob = self.actor(input)
        
        return state_value_function, action_prob


def train(env, agent, optimizer, num_episodes, scheduler = None, gamma=0.99):
    running_avg_le = []
    all_rewards = []

    for episode in range(num_episodes):
        log_probs = []
        values = []
        rewards = []

        obs = env.reset()
        done = False
        optimizer.zero_grad()

        while not done:
            value, action_logit = agent(obs)
            action_prob = Categorical(action_logit)
            action = action_prob.sample()

            log_prob = action_prob.log_prob(action)
            obs, reward, done, _ = env.step(action.item())

            rewards.append(reward)
            log_probs.append(action_prob.log_prob(action))
            values.append(value)
        print(sum(rewards))
        values.append(torch.FloatTensor([0]))

        all_rewards.append(sum(rewards))
        rewards = np.asarray(rewards)
        rewards = (rewards - rewards.mean())/rewards.std()

        if (episode+1) % 50 == 0:
            if episode >= 100:
                print('Episode: {}, Reward: {}.'.format(episode, sum(all_rewards[-100:])/100))
            else:
                print('Episode: {}, Reward: {}.'.format(episode, sum(all_rewards)/len(all_rewards)))
        
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)
        
        adv = torch.zeros_like(values[:-1])
        Q_val = torch.zeros_like(adv)

        for t in reversed(range(len(rewards))):
            Q_val[t] = rewards[t] + gamma * values[t+1].detach()

        adv = Q_val - values[:-1]

        loss_actor = (-log_probs * adv.detach()).sum()
        loss_critic = (adv**2).sum()
        total_loss = loss_actor + loss_critic

        total_loss.backward()
        optimizer.step()


def eval(env, agent, num_episode):
    all_rewards = []

    for i in range(num_episode):
        obs = env.reset()
        done =False
        ep_reward = 0

        while not done:
            env.render()
            _, action_logits = agent(obs)
            action = torch.argmax(action_logits).item()

            obs, reward, done, _ = env.step(action)
            ep_reward += reward

        obs = env.reset()
        all_rewards.append(ep_reward)
    
    episodes = np.arange(num_episode+1)

    plt.plot(episodes, all_rewards, 'g', label='Training accuracy')
    plt.title('Evaluation Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    

env = gym.make('LunarLander-v2')

NUM_INPUTS = env.observation_space.shape[0]
NUM_HIDDEN = 128
NUM_OUTPUTS = env.action_space.n
LEARNING_RATE = 3e-3

actor = backbone('actor', NUM_INPUTS, NUM_HIDDEN, NUM_OUTPUTS)
critic = backbone('critic', NUM_INPUTS, NUM_HIDDEN, 1)
agent = policy(actor, critic)

optimizer = RMSprop(agent.parameters(), lr = LEARNING_RATE)

train(env, agent, optimizer, 10_000)