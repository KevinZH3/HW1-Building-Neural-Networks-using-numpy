import itertools
from tqdm import tqdm
from src.LoadData import LoadDataset
from src.ModelTrainer import Trainer
from src.NeuralNetwork import NeuralNetwork
from utils.Optimizer import SGDOptimizer
from utils.CrossEntropyLoss import CrossEntropyLoss


class SearchParam:
    def __init__(self, params_grids, params_init):
        self.res = []
        self.grids = self.gen_grids(params_grids, params_init)

    @staticmethod
    def gen_grids(hyper_param_grids, hyper_param_init):
        for key in hyper_param_grids.keys():
            if len(hyper_param_grids[key]) == 0:
                hyper_param_grids.pop(key)
        for key in hyper_param_init.keys():
            if key not in hyper_param_grids.keys() or len(hyper_param_grids[key]) == 0:
                hyper_param_grids[key] = [hyper_param_init[key]]  

        grids = []
        for values in itertools.product(*hyper_param_grids.values()):
            grid = dict(zip(hyper_param_grids.keys(), values))
            grids.append(grid)
        return grids

    @staticmethod
    def gen_architecture(grid):
        optimizer_params = {
            "lr": grid["lr"],
            "l2": grid["l2"],
            "decay_rate": grid["decay_rate"],
            "decay_step": grid["decay_step"],
        }

        n_layers = sum([1 for key in grid.keys() if "hidden_size" in key]) + 1
        architecture = []
        if n_layers == 1:
            layer = {
                "dim_in": 784,
                "dim_out": 10,
                "act_func": grid["activation_1"],
            }
            architecture.append(layer)
        elif n_layers > 1:
            layer = {
                "dim_in": 784,
                "dim_out": grid["hidden_size_1"],
                "act_func": grid["activation_1"],
            }
            architecture.append(layer)
            for i in range(1, n_layers - 1):
                layer = {
                    "dim_in": grid[f"hidden_size_{i}"],
                    "dim_out": grid[f"hidden_size_{i + 1}"],
                    "act_func": grid[f"activation_{i + 1}"],
                }
                architecture.append(layer)
            layer = {
                "dim_in": grid[f"hidden_size_{n_layers - 1}"],
                "dim_out": 10,
                "act_func": grid[f"activation_{n_layers}"],
            }
            architecture.append(layer)
        return architecture, optimizer_params

    def search(self, dataloader_params, trainer_params):
        for grid in tqdm(self.grids):
            architecture, optimizer_params = self.gen_architecture(grid)
            dataloader = LoadDataset(**dataloader_params)
            model = NeuralNetwork(architecture)
            optimizer = SGDOptimizer(**optimizer_params)
            loss = CrossEntropyLoss()

            trainer = Trainer(model, optimizer, loss, dataloader, **trainer_params)
            trainer.train(save_model=False, verbose=False)
            valid_loss, valid_acc = trainer.evaluate()
            self.res.append((grid, valid_loss, valid_acc))
            self.res.sort(key=lambda x: x[1])

        return self.res
