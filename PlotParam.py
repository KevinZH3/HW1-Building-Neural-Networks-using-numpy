import json
import matplotlib.pyplot as plt
from src.NeuralNetwork import Linear
from src.NeuralNetwork import NeuralNetwork

model_path = "./model/model_saved.pkl"
architecture = json.load(open(model_path.replace(".pkl", ".json"), "r"))

model = NeuralNetwork(architecture)
for i, layer in enumerate(model.layers):
    if isinstance(layer, Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=50)
        plt.title(f"Initial Weight Distribution of Layer {i + 1}")
        plt.show()

model.load_model(path=model_path)
for i, layer in enumerate(model.layers):
    if isinstance(layer, Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=50)
        plt.title(f"Weight Distribution of Layer {i + 1} ")
        plt.show()

