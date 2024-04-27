import json
import pickle
import numpy as np


class ActFunc:
    def __init__(self, activation_type):
        self.input_cache = None
        self.type = activation_type

    def forward(self, z):
        self.input_cache = z
        if self.type == "relu":
            return np.maximum(0, z)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.type == "softmax":
            exps = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exps / np.sum(exps, axis=1, keepdims=True)

    def backward(self, dA, z):
        if self.type == "sigmoid" or self.type == "softmax":
            sig = self.forward(z)
            return dA * sig * (1 - sig)
        elif self.type == "relu":
            return dA * (z > 0)


class Linear:
    def __init__(self, dim_in, dim_out):
        self.input_cache = None
        self.b = np.zeros((1, dim_out))
        self.W = np.random.normal(0, pow(dim_in, -0.5), (dim_in, dim_out))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.input_cache = x
        return np.dot(x, self.W) + self.b

    def backward(self, dZ, x):
        db = np.sum(dZ, axis=0, keepdims=True)
        dW = np.dot(x.T, dZ)
        dx = np.dot(dZ, self.W.T)
        return dW, db, dx

    def zero_grad(self):
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)


class NeuralNetwork:
    def __init__(self, architecture=None):
        self.layers = []
        self.architecture = architecture if architecture else []
        for layer in self.architecture:
            self.layers.append(Linear(layer["dim_in"], layer["dim_out"]))
            self.layers.append(ActFunc(layer["act_func"]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                dW, db, grad = layer.backward(grad, layer.input_cache)
                layer.dW = dW
                layer.db = db
            elif isinstance(layer, ActFunc):
                grad = layer.backward(grad, layer.input_cache)
        return grad

    def copy_model(self):
        model_cp = NeuralNetwork(self.architecture)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                model_cp.layers[i].W = layer.W.copy()
                model_cp.layers[i].b = layer.b.copy()
        return model_cp

    def save_model(self, path):
        model_weight = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                model_weight[f"layer_{i}_W"] = layer.W
                model_weight[f"layer_{i}_b"] = layer.b
        with open(path, "wb") as f:
            pickle.dump(model_weight, f)
        with open(path.replace(".pkl", ".json"), "w") as f:
            json.dump(self.architecture, f)

    def load_model(self, path):
        with open(path.replace(".pkl", ".json"), "r") as f:
            architecture = json.load(f)
        self.__init__(architecture)
        with open(path, "rb") as f:
            model_weight = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                layer.W = model_weight[f"layer_{i}_W"]
                layer.b = model_weight[f"layer_{i}_b"]
                layer.zero_grad()
    
    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)
