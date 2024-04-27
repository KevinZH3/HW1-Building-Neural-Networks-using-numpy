import json
import matplotlib.pyplot as plt
from src.NeuralNetwork import Linear
from src.NeuralNetwork import NeuralNetwork

model_path = "./model/model_saved.pkl"
architecture = json.load(open(model_path.replace(".pkl", ".json"), "r"))

model = NeuralNetwork(architecture)
j = 1
for _, layer in enumerate(model.layers):
    if isinstance(layer, Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=50)
        plt.title(f"Initial Weight Distribution of Layer {j}")
        plt.show()
        j += 1

model.load_model(path=model_path)
j = 1
for _, layer in enumerate(model.layers):
    if isinstance(layer, Linear):
        plt.figure()
        plt.hist(layer.W.flatten(), bins=50)
        plt.title(f"Weight Distribution of Layer {j} ")
        plt.show()
        j += 1

