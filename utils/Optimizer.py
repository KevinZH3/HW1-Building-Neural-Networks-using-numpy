from src.NeuralNetwork import Linear

class SGDOptimizer:
    def __init__(self, lr, l2=0, decay_rate=0, decay_step=1000):
        self.lr = lr
        self.l2 = l2
        self.decay_rate = decay_rate if 0 < decay_rate < 1 else None
        self.decay_step = decay_step if 0 < decay_rate < 1 else None
        self.iterations = 0

    def step(self, model):
        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.W -= self.lr * (layer.dW + self.l2 * layer.W)
                layer.b -= self.lr * layer.db
                layer.zero_grad()
        self.iterations += 1
        if self.decay_rate and self.iterations % self.decay_step == 0:
            self.lr *= self.decay_rate if self.lr > 0.01 else 1
