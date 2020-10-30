import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class FrameAnalyze():
    def __init__(self, input_dims,output_dims, batch_size=32, conv1_dims=32, conv2_dims=64,layer1_dims=256, layer2_dims=128, layer3_dims=64, mem_size=100000, fname="flappy_cnn.h5"):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.layer3_dims = layer3_dims
        self.conv1_dims = conv1_dims
        self.conv2_dims = conv2_dims
        self.model_file = fname
        self.batch_size = batch_size

        self.mem_size = mem_size
        self.mem_counter = 0
        self.memory_image = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.memory_values = np.zeros((self.mem_size, *output_dims), dtype=np.float32)
        self.history = []

    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv3D(self.conv1_dims, kernel_size=(8,8,1), strides=1, activation='relu', input_shape=(*input_dims, 1)),
            keras.layers.MaxPooling3D(pool_size=(2, 2, 1)),
            keras.layers.Conv3D(self.conv2_dims, kernel_size=(3,3,1), strides=2, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(self.layer1_dims, 'relu'),
            keras.layers.Dense(self.layer2_dims, 'relu'),
            keras.layers.Dense(self, self.layer3_dims, 'relu'),
            keras.layers.Dense(self.output_dims)
        ])

    def store(self, input, values):
        memory_index = self.mem_counter % self.mem_size
        self.memory_image[memory_index] = input
        self.memory_values[memory_index] = values
        self.mem_counter += 1

    def predict_on_image(self, input):
        return self.model.predict(input)
    
    def sampleStorage(self):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        stacks = self.memory_image[batch]
        values = self.memory_values[batch]
        return stacks, values

    def learn_vision(self):
        if self.mem_counter < self.batch_size:
            return
        stacks, values = self.sampleStorage()
        self.history.append(self.model.train_on_batch(stacks, values))
    
    def save_model(self):
        print ("Saving CNN...")
        self.model.save(self.model_file)
        print ("CNN Saved to ", self.model_file)

    def load_model(self):
        self.model = keras.models.load_model(self.model_file)
        print ("Retrieved CNN from...", self.model_file)
        
