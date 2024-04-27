import matplotlib.pyplot as plt
from src.LoadData import LoadDataset
from utils.CrossEntropyLoss import CrossEntropyLoss
from src.NeuralNetwork import NeuralNetwork
from utils.Optimizer import SGDOptimizer
from src.ModelTrainer import Trainer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--valid_size', '-vs', type=int, default=1000)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--lr', '-lr', type=int, default=0.01)
    parser.add_argument('--l2', '-l2', type=int, default=0.001)
    parser.add_argument('--decay_rate', '-dr', type=int, default=0.95)
    parser.add_argument('--decay_step', '-ds', type=int, default=5000)
    args = parser.parse_args()
    
    dataloader = LoadDataset(path='data', valid_size=args.valid_size, batch_size=args.batch_size) 
    model = NeuralNetwork(
        [
            {"dim_in": 784, "dim_out": 128, "act_func": "relu"},
            {"dim_in": 128, "dim_out": 32, "act_func": "relu"},
            {"dim_in": 32, "dim_out": 10, "act_func": "softmax"},
        ]
    )
    optimizer = SGDOptimizer(lr=args.lr, l2=args.l2, decay_rate=args.decay_rate, decay_step=args.decay_step)  
    loss = CrossEntropyLoss()  

    trainer = Trainer(model, optimizer, loss, dataloader, n_epochs=args.epochs)  
    trainer.train(save_model=True, verbose=True)  # 训练模型
    trainer.save_best_model("model/", best_cnt=3)  # 保存最优模型
    trainer.model_cache = {}

    plt.show(block=True)
