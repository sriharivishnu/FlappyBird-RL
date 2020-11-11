import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
class FrameStacker():
    def __init__(self, input_dims, stack_size=4):
        self.stack_size = stack_size
        self.buffer = np.zeros((self.stack_size, *input_dims))
    def add(self, img):
        for i in range(self.stack_size - 1):
            self.buffer[i] = self.buffer[i+1]
        self.buffer[self.stack_size - 1] = img
    
    def getStack(self):
        return self.buffer

class FrameAnalyze():
    def __init__(self, input_dims,output_dims, batch_size=32, conv1_dims=32, conv2_dims=64,layer1_dims=256, layer2_dims=128, layer3_dims=64, mem_size=100000, fname="flappy_cnn.h5", process_rate=10):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.layer3_dims = layer3_dims
        self.conv1_dims = conv1_dims
        self.conv2_dims = conv2_dims
        self.model_file = fname
        self.batch_size = batch_size
        self.model = self.build_model()

        self.counter = 0
        self.process_rate = process_rate

        self.mem_size = mem_size
        self.mem_counter = 0
        self.memory_image = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.memory_values = np.zeros((self.mem_size, *output_dims), dtype=np.float32)
        self.history = []

        self.frame_stacker = FrameStacker((128, 76))

    def build_model(self):
        model =  keras.Sequential([
            keras.layers.Conv3D(self.conv1_dims, kernel_size=(1,7,7), strides=2, activation='relu', input_shape=(*self.input_dims, 1)),
            keras.layers.MaxPooling3D(pool_size=(1, 2, 2)),
            keras.layers.Conv3D(self.conv2_dims, kernel_size=(1,3,3), strides=2, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(self.layer1_dims, 'relu'),
            keras.layers.Dense(self.layer2_dims, 'relu'),
            keras.layers.Dense(self.layer3_dims, 'relu'),
            keras.layers.Dense(self.output_dims[0])
        ])
        model.compile(optimizer='Adam', loss='mean_squared_error')
        print ("CNN SUMMARY")
        print (model.summary())
        return model

    def store(self, input, values):
        if self.counter % self.process_rate < 4:
            self.frame_stacker.add(input)
        if self.counter % self.process_rate != 0:
            return
        memory_index = self.mem_counter % self.mem_size
        self.memory_image[memory_index] = self.frame_stacker.getStack()
        self.memory_values[memory_index] = values
        self.mem_counter += 1

    def predict_on_frames(self, input):
        return self.model.predict(input)
    
    def sampleStorage(self):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        stacks = self.memory_image[batch]
        values = self.memory_values[batch]
        return stacks, values

    def learn_vision(self):
        self.counter += 1  
        if self.counter % self.process_rate != 0:
            return
        if self.mem_counter < self.batch_size:
            return
        stacks, values = self.sampleStorage()
        self.history.append(self.model.train_on_batch(np.expand_dims(stacks, axis=4), values))
        print ("LOSS: ", self.history[-1])
    
    def save_model(self):
        print ("Saving CNN...")
        self.model.save(self.model_file)
        print ("CNN Saved to ", self.model_file)

    def load_model(self):
        self.model = keras.models.load_model(self.model_file)
        print ("Retrieved CNN from...", self.model_file)
        
