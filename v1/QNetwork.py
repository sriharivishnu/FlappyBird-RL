import numpy as np
from tensorflow import keras
import pickle
import os
import random
class ReplayBuffer():
    def __init__(self, max_size, input_dims, fname="dqn_replay_buffer.pickle"):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

        self.fname = fname
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
    
    def save(self):
        values = {}
        values['state_memory'] = self.state_memory
        values['new_state_memory'] = self.new_state_memory
        values['reward_memory'] = self.reward_memory
        values['action_memory'] = self.action_memory
        values['terminal_memory'] = self.terminal_memory
        values['count'] = self.mem_cntr
        with open(self.fname, 'wb') as fp:
            pickle.dump(values, fp)
    
    def load(self):
        if os.path.exists(self.fname):
            with open(self.fname, 'rb') as fp:
                values = pickle.load(fp)
                self.state_memory = values['state_memory']
                self.new_state_memory = values['new_state_memory']
                self.reward_memory = values['reward_memory']
                self.action_memory = values['action_memory']
                self.terminal_memory = values['terminal_memory']
                self.mem_cntr = values['count']
    
def build_dqn(lr, action_dim, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims):
    model = keras.Sequential([
        keras.layers.Dense(fc1_dims, activation='relu', input_shape=input_dims),
        keras.layers.Dense(fc2_dims, activation='relu'),
        keras.layers.Dense(fc3_dims, activation='relu')
    ])
    if (fc4_dims > 0):
        model.add(keras.layers.Dense(fc4_dims, activation='relu'))
    model.add(keras.layers.Dense(action_dim))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')
    print(model.summary())
    return model
"""
V3, V4: 128, 64, 32
V6: 128, 64, 32, 4
V7: 512, 256, 32, 0 (Best)
"""
class Agent():
    def __init__(self, lr, gamma, action_dim, epsilon, batch_size,
    input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000, fname='dqn_model_flappy_V7.h5',
    fc1_dims=128, fc2_dims=64, fc3_dims=32, fc4_dims=4, replace=100):
        self.action_space = [i for i in range(action_dim)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.eps_min = epsilon_end 
        self.replace = replace

        self.batch_size = batch_size
        self.model_file = "models/"+fname 

        self.learn_count = 1
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, action_dim, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims)
        self.q_next = build_dqn(lr, action_dim, input_dims, fc1_dims, fc2_dims, fc3_dims, fc4_dims)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            if random.random() > 0.96:
                action = 1
            else:
                action = 0
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        if self.learn_count % self.replace == 0:
            self.update_target_network()
        
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_eval.predict_on_batch(states)
        q_next = self.q_next.predict_on_batch(states_)
        q_target = np.copy(q_pred)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1)*dones
        self.learn_count += 1

        self.epsilon = self.epsilon - self.epsilon_dec if self.epsilon > self.eps_min else self.eps_min
        return self.q_eval.train_on_batch(states, q_target)
    
    def update_target_network(self):
        tau = 0.7
        next = self.q_next.get_weights()
        eval = self.q_eval.get_weights()
        new_weights = []
        for x in range(len(next)):
            new_weights.append(next[x] * (1-tau) + eval[x] * tau)
        
        # new_weights = self.q_next.get_weights() * (1-tau) + self.q_eval.get_weights() * tau
        self.q_next.set_weights(new_weights)

    def save_model(self):
        self.q_eval.save(self.model_file)
        self.memory.save()

    def load_model(self):
        if (os.path.exists(self.model_file)):
            self.q_eval = keras.models.load_model(self.model_file)
            self.q_next = keras.models.load_model(self.model_file)
            self.epsilon = 0.04
        self.memory.load()