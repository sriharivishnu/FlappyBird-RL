import numpy as np
from tensorflow import keras
import random
import pickle as pkl
import os

class Agent():
    def __init__(self, lr, gamma, action_dim, epsilon, batch_size,
    input_dims, epsilon_dec=1e-3, epsilon_end=0.001, mem_size=1000000, fname='dqn_model_flappy_V4.h5',
    fc1_dims=64, fc2_dims=32, fc3_dims=32, replace=100):
        self.action_space = [i for i in range(action_dim)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.eps_min = epsilon_end 
        self.lr = lr
        self.q_table = {}
        self.game_history = []
        self.moves = []
        self.mem_cntr = 0
        self.load()
        
        # self.memory = ReplayBuffer(mem_size, input_dims)
        # self.q_eval = build_dqn(lr, action_dim, input_dims, fc1_dims, fc2_dims, fc3_dims)
        # self.q_next = build_dqn(lr, action_dim, input_dims, fc1_dims, fc2_dims, fc3_dims)
    def string_from_state(self, state):
        """
        ypos_yvel_xPipe_yPipe_nextPipe_
        """
        res = ""
        for x in range(len(state)):
            res += str(state[x])+"_"
        return res

    def store_transition(self, state, action, reward, new_state, done):
        string_state =  self.string_from_state(state)
        new_string_state = self.string_from_state(new_state)
        self.check(string_state)
        self.moves.append((string_state, action, reward, new_string_state))
        self.mem_cntr = len(self.q_table)

    def updateQ(self):
        self.moves.reverse()
        for move in self.moves:
            state, action, reward, state_ = move
            self.q_table[state][action] = (1-self.lr) * self.q_table[state][action] \
            + self.lr * (reward + self.gamma * np.max(self.get_state(state_)))
        self.moves = []

    def get_state(self, state):
        self.check(state) 
        return self.q_table[state]
        
    def check(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.array([0,0], dtype=np.float32)
            return False
        return True

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            if random.random() > 0.9:
                action = 1
            else:
                action = 0
        else:
            state = self.string_from_state(observation)
            self.check(state)
            actions = self.q_table[state]
            if actions[0] == actions[1]:
                action = 0
            else:
                action = np.argmax(actions)

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.eps_min else self.eps_min
        return action
    
    def save(self):
        with open('q_values.pickle', 'wb') as fp:
            pkl.dump(self.q_table, fp)
    
    def load(self):
        if os.path.exists('q_values.pickle'):
            with open('q_values.pickle', 'rb') as fp:
                self.q_table = pkl.load(fp)
    
    # def learn(self):
    #     if self.memory.mem_cntr < self.batch_size:
    #         return
        
    #     if self.learn_count % self.replace == 0:
    #         self.q_next.set_weights(self.q_eval.get_weights())
        
    #     states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
    #     q_pred = self.q_eval.predict_on_batch(states)
    #     q_next = self.q_next.predict_on_batch(states_)
    #     q_target = np.copy(q_pred)
    #     batch_index = np.arange(self.batch_size, dtype=np.int32)

    #     q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*dones
    #     self.learn_count += 1

    #     self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.eps_min else self.eps_min
    #     return self.q_eval.train_on_batch(states, q_target)
    
    # def update_target_network(self):
    #     self.target_network.set_weights(self.q_eval.get_weights())

    # def save_model(self):
    #     self.q_eval.save(self.model_file)

    # def load_model(self):
    #     self.q_eval = keras.models.load_model(self.model_file)
    #     self.q_next = keras.models.load_model(self.model_file)