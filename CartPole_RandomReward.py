import gym
import numpy as np

def run_episode(env, parameters):
    observation = env.reset()
    totalReward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalReward += reward
        if done:
            break
    return totalReward

def train(submit):
    env = gym.make('CartPole-v0')
    if submit:
        env.monitor.start('cartpole-experiments/', force=True)
    count = 0
    bestparams = 0
    bestreward = 0
    for _ in range(10000):
        count += 1
        parameters = np.random.rand(4) * 2 - 2
        reward = run_episode(env, parameters)
        if reward > bestreward:
            bestreward = reward
            bestparams = parameters
            if reward == 200:
                break
    if submit:
        for _ in range(100):
            run_episode(env, bestparams)
        env.monitor.close()
    return count

env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
