# coding: utf-8

import numpy as np

"""
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in xrange(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
"""
"""
import gym
env = gym.make('MountainCar-v0')
for i_episode in xrange(1):
    observation = env.reset()
    print observation
    for t in xrange(200):
        env.render()
        action = env.action_space.sample()
        print action
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break

import gym
env = gym.make('MountainCar-v0')
print np.shape(np.array(env.observation_space.high))
print env.observation_space.low
print env.action_space.n
print env.shape(env.observation_space)

from itertools import combinations

original = "abc"
print combinations(original, 2)
"""

import gym
env = gym.make('CartPole-v0')
env.monitor.start('/tmp/cartpole-experiment-1',force=True)
for i_episode in xrange(200):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break

env.monitor.close()
