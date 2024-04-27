import numpy as np
import gzip
import os

class LoadDataset:
    def __init__(self, path ="data", valid_size=1000, batch_size=32):
        self.batch_size = batch_size

        label_file_path = os.path.join(path, 'train-labels-idx1-ubyte.gz')
        with gzip.open(label_file_path, 'rb') as f:
            y = np.frombuffer(f.read(), np.uint8, offset=8)
            y = np.eye(10)[y] 

        image_file_path = os.path.join(path, 'train-images-idx3-ubyte.gz')
        with gzip.open(image_file_path, 'rb') as f:
            X = np.frombuffer(f.read(), np.uint8, offset=16)
            X = X.reshape(-1, 784).astype(np.float32) / 255.0
        self.X_train, self.y_train, self.X_valid, self.y_valid = self.split_train_valid(X, y, valid_size)

        label_file_path = os.path.join(path, 't10k-labels-idx1-ubyte.gz')
        with gzip.open(label_file_path, 'rb') as f:
            self.y_test = np.frombuffer(f.read(), np.uint8, offset=8)
            self.y_test = np.eye(10)[self.y_test]

        image_file_path = os.path.join(path, 't10k-images-idx3-ubyte.gz')
        with gzip.open(image_file_path, 'rb') as f:
            self.X_test = np.frombuffer(f.read(), np.uint8, offset=16)
            self.X_test = self.X_test.reshape(-1, 784).astype(np.float32) / 255.0
        
    @staticmethod
    def split_train_valid(X_train, y_train, valid_size):
        sample_size = X_train.shape[0]
        indices = np.random.permutation(sample_size)
        valid_indices = indices[:valid_size]
        train_indices = indices[valid_size:]
        return X_train[train_indices], y_train[train_indices], X_train[valid_indices], y_train[valid_indices]

    def gen_batch(self, data='train'):
        if data == 'train':
            n_samples = self.X_train.shape[0]
            indices = np.random.permutation(self.X_train.shape[0])
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield self.X_train[batch_indices], self.y_train[batch_indices]
        elif data == 'valid':
            n_samples = self.X_valid.shape[0]
            indices = np.arange(n_samples)
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield self.X_valid[batch_indices], self.y_valid[batch_indices]
        elif data == 'test':
            n_samples = self.X_test.shape[0]
            indices = np.arange(n_samples)
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                yield self.X_test[batch_indices], self.y_test[batch_indices]
