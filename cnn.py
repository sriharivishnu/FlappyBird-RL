import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class FrameAnalyze():
    def __init__(self, output_dims, conv1_dims=32, conv2_dims=64,layer1_dims=256, layer2_dims=128, fname="flappy_cnn.h5"):
        self.output_dims = output_dims
        self.layer1_dims = layer1_dims
        self.layer2_dims = layer2_dims
        self.conv1_dims = conv1_dims
        self.conv2_dims = conv2_dims
        self.model_file = fname
        self.history = []

    def build_model(self):
        self.model = keras.Sequential([
            keras.layers.Conv3D(self.conv1_dims, kernel_size=8, strides=4, activation='relu'),
            keras.layers.Conv3D(self.conv2_dims, kernel_size=4, strides=2, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(self.layer1_dims, 'relu'),
            keras.layers.Dense(self.layer2_dims, 'relu'),
            keras.layers.Dense(self.output_dims)
        ])

    def predict_on_image(self, input):
        return self.model.predict(input)

    def train_on_image(self, input, actual_values):
        actual = np.array(actual_values)
        self.history.append(self.model.train_on_batch(input, actual))
    
    def save_model(self):
        self.model.save(self.model_file)

    def load_model(self):
        self.model = keras.models.load_model(self.model_file)
        
