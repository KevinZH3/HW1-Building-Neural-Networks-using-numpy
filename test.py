import numpy as np
from src.LoadData import LoadDataset
from utils.CrossEntropyLoss import CrossEntropyLoss
from src.NeuralNetwork import NeuralNetwork

weight_path = "model/model_saved.pkl"
if __name__ == "__main__":
    dataloader = LoadDataset(path='data', batch_size=32)
    model = NeuralNetwork()
    model.load_model(weight_path)  # 从已经训练好的权重加载模型
    loss = CrossEntropyLoss()

    total_loss = 0
    total_acc = 0
    for X_batch, y_batch in dataloader.gen_batch(data='test'):
        y_pred = model.forward(X_batch)
        total_acc += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        ce_loss = loss.forward(y_pred, y_batch)
        total_loss += ce_loss * len(X_batch)

    test_loss = total_loss / len(dataloader.y_test)
    test_acc = total_acc / len(dataloader.y_test)

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
