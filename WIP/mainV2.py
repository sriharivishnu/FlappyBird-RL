from fastV2 import FlappyGame
from random import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import gym
import cv2
from QNetwork import Agent

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    env = FlappyGame()
    lr = 0.7
    n_games = 25000
    agent = Agent(gamma=0.9, epsilon=1.0, lr=lr, input_dims=(4,), action_dim=2, mem_size=1000000, batch_size=1024,
    epsilon_end=0.01, epsilon_dec=1e-5)
    agent.epsilon = 0.2
    # if (os.path.exists(agent.model_file)):
    #     agent.load_model()
    #     agent.epsilon = 0.2
    #     agent.eps_min = 0.01
    #     print ("Loaded model from ", agent.model_file)

    # Initialize scores and epsilon history
    scores = []
    eps_hist = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        total_reward = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, time, score = env.step(action)
            total_reward += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
        
        agent.updateQ()
        eps_hist.append(agent.epsilon)
        scores.append(env.score)

        avg_score = np.mean(scores[-100:])
        max_score = np.max(scores[-100:])
        if (i % 100 == 0):
            print ('episode: ', i, '| score %.2f' % env.score, 
                    '| average score %.2f' % avg_score,
                    '| reward for episode: ', total_reward,
                    '| epsilon %.2f' % agent.epsilon,
                    '| mem_cntr', agent.mem_cntr,
                    '| max_score', max_score)
    
    # print ("Saving model to'", agent.model_file, "'. Please wait...")
    agent.save()
    print ("Saved Models")
    # plt.title("Cost vs Batches")
    # plt.show()

    

