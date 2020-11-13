from fast import FlappyGame
from random import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import gym
import cv2
from cnn import FrameAnalyze
from QNetwork import Agent
"""
playerPosY, playerVelY, nextPipeX, nextPipeY, NextnextPipeX, nextNextPipeY])
"""
def preprocessObservation(observation : np.array, game : FlappyGame):
    minX = 0
    minY = 0
    minVelY = game.playerMinVelY
    
    maxX = game.SCREENWIDTH * 3
    maxY = game.BASEY
    maxVelY = game.playerMaxVelY
    return observation
    # mins = np.array([minY, minVelY, minX, minY, minX, minY])
    # maxes = np.array([maxY, maxVelY, maxX, maxY, maxX, maxY])
    # return (observation - mins) / (maxes - mins)

if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    env = FlappyGame()
    lr = 0.001
    n_games = 1000
    agent = Agent(gamma=0.99, epsilon=1.0, lr=lr, input_dims=(7,), action_dim=2, mem_size=1000000, batch_size=128,
    epsilon_end=0.003, epsilon_dec=1e-5, fname='dqn_model_flappy_V6.h5', 
    fc1_dims=128, fc2_dims=64, fc3_dims=32, fc4_dims=16, replace=1000)
    
    # frame_analyze = FrameAnalyze((4,128,76), (8,))
    if (os.path.exists(agent.model_file)):
        agent.load_model()
        agent.epsilon = 0.01
        # agent.eps_min = 0.00
        print ("Loaded model from ", agent.model_file)
    # if (os.path.exists(frame_analyze.model_file)):
    #     frame_analyze.load_model()
    #     print ("Loaded CNN from ", frame_analyze.model_file)

    # Initialize scores and epsilon history
    scores = []
    eps_hist = []
    for i in range(n_games):
        done = False
        observation = preprocessObservation(env.reset(), env)
        total_reward = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, time, score = env.step(action)
            observation_ = preprocessObservation(observation_, env)
            total_reward += reward
            agent.store_transition(observation, action, reward, observation_, done)
            # print (observation)
            # cv2.imshow("Cool", env._getScreen())
            # frame_analyze.store(env._getScreen(), observation_)
            agent.learn()
            # frame_analyze.learn_vision()
            observation = observation_
        
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
    # frame_analyze.save_model()
    print ("Saved Models")
    # plt.title("Cost vs Batches")
    # plt.plot(list(range(len(frame_analyze.history))), frame_analyze.history)
    plt.show()

    

