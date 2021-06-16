import gym
import torch
from train import train
from model import fc

env = gym.make('CartPole-v1')
agent = fc(env.observation_space.shape[0], env.action_space.n)
learning_rate = 0.01
num_ep = 10
EPOCHS = 50

def main():
    for epoch in range(EPOCHS):
        avg_reward = train(env, agent, num_ep, learning_rate)
        print('For epoch {}, the average reward received was {}'.format(epoch, avg_reward))
    
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action = torch.argmax(agent.forward(obs))
        obs, _, done, _ = env.step(int(action))
        if done:
            done = False
            obs = env.reset()

if __name__ == '__main__':
    main()