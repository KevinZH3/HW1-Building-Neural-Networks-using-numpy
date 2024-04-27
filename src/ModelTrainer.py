import os
import time
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, loss, dataloader, n_epochs):
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dataloader = dataloader
        self.n_epochs = n_epochs
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.model_cache = {}


    def train(self, save_model=True, verbose=True):
        for epoch in range(1, self.n_epochs + 1):
            start = time.time()
            total_loss = 0
            total_acc = 0
            for X_batch, y_batch in self.dataloader.gen_batch(data='train'):
                y_pred = self.model.forward(X_batch)
                total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
                loss = self.loss.forward(y_pred, y_batch)
                total_loss += loss * len(X_batch)
                grad = self.loss.backward()
                self.model.backward(grad)
                self.optimizer.step(self.model)
            end = time.time()

            train_loss = total_loss / len(self.dataloader.y_train)
            train_acc = total_acc / len(self.dataloader.y_train)
            self.train_loss.append(train_loss)
            self.train_acc.append(train_acc)

            valid_loss, valid_acc = self.evaluate()
            self.valid_loss.append(valid_loss)
            self.valid_acc.append(valid_acc)

            if save_model:
                self.model_cache[epoch] = self.model.copy_model()

            f = len(str(self.n_epochs))
            if verbose:
                self.plot_loss(epoch)
                print(f"Epoch {epoch:0>{f}} | "
                    f"learning rate: {self.optimizer.lr:.4f} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f} | "
                    f"Epoch Time: {end - start:.2f} s")

        if save_model:
            self.model_cache[self.n_epochs] = self.model.copy_model()


    def evaluate(self):
        total_loss = 0
        total_acc = 0
        for X_batch, y_batch in self.dataloader.gen_batch(data='valid'):
            y_pred = self.model.forward(X_batch)
            total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
            loss = self.loss.forward(y_pred, y_batch)
            total_loss += loss * len(X_batch)
        valid_loss = total_loss / len(self.dataloader.y_valid)
        valid_acc = total_acc / len(self.dataloader.y_valid)
        return valid_loss, valid_acc


    def save_best_model(self, path, best_cnt=1):
        """
        根据每个epoch训练中测试集的loss指标, 保存最好的几组模型权重
        """
        best_idxs = np.argsort(self.valid_loss)[:best_cnt]
        best_epochs = [(index + 1) for index in best_idxs]
        for epoch in best_epochs:
            if epoch not in self.model_cache:
                continue
            model = self.model_cache[epoch]
            model.save_model(os.path.join(path, "model_saved.pkl"))


    def plot_loss(self, epoch):
        """
        可视化展示训练过程中在训练集和测试集上的 loss 和 acc
        """
        plt.figure(1)
        plt.clf()
        ax1 = plt.gca()

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        line1, = ax1.plot(range(1, epoch + 1), self.train_loss, color='red', label='Train Loss')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuray')
        line2, = ax2.plot(range(1, epoch + 1), self.train_acc, color='blue', label='Train Acc', linestyle = '-.')
        ax2.tick_params(axis='y')

        line3, = ax1.plot(range(1, epoch + 1), self.valid_loss, color='green', label='Valid Loss')

        line4, = ax2.plot(range(1, epoch + 1), self.valid_acc, color='gold', label='Valid Acc', linestyle = '-.')

        lines = [line1, line2, line3, line4]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
