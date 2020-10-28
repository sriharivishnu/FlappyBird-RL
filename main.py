from flappy import FlappyGame
from random import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import gym
from QNetwork import Agent
class FrameStacker():
    def __init__(self, input_dims, stack_size=4):
        self.stack_size = stack_size
        self.buffer = np.zeros((*input_dims, self.stack_size))
    def add(self, img):
        for i in range(self.stack_size - 1):
            self.buffer[:,:, i] = self.buffer[:,:, i+1]
        self.buffer[:,:, self.stack_size - 1] = img
    
    def getStack(self):
        return self.buffer

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    env = FlappyGame()
    lr = 0.001
    n_games = 500
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=(5,), action_dim=2, mem_size=1000000, batch_size=64,
    epsilon_end=0.01, epsilon_dec=1e-3)
    if (os.path.exists(agent.model_file)):
        agent.load_model()
        agent.epsilon = 0.01
        print ("Loaded model from ", agent.model_file)

    scores = []
    eps_hist = []
    # frames = FrameStacker((128,76))
    for i in range(n_games):
        done = False
        # frames.add(env.reset())
        # observation = frames.getStack()
        observation = env.reset()
        total_reward = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, time, score = env.step(action)
            # print(reward)
            # frames.add(observation_)
            total_reward += reward
            # observation_ = frames.getStack()
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        eps_hist.append(agent.epsilon)
        scores.append(env.score)

        avg_score = np.mean(scores[-100:])
        print ('episode: ', i, '| score %.2f' % env.score, 
                '| average score %.2f' % avg_score,
                '| reward for episode: ', total_reward,
                '| epsilon %.2f' % agent.epsilon,
                '| mem_cntr', agent.memory.mem_cntr)
    
    print ("Saving model to'", agent.model_file, "'. Please wait...")
    agent.save_model()
    print ("Saved Model")

