import json
from src.SearchParam import SearchParam


if __name__ == "__main__":
    # 供搜索的超参数网格
    params_grids = {
        "hidden_size_1": [128, 256],
        "hidden_size_2": [64, 32],
        "lr": [0.05, 0.01],
        "l2": [0.001, 0.005],
    }

    # 超参数搜索的初始值
    params_init = {
    "dim_in": 784,
    "hidden_size_1": 128,
    "hidden_size_2": 32,
    "dim_out": 10,
    "activation_1": "relu",
    "activation_2": "relu",
    "activation_3": "softmax",
    "lr": 0.05,
    "l2": 0.001,
    "decay_rate": 0.95,
    "decay_step": 6000,
    }  
    searcher = SearchParam(params_grids, params_init)
    trainer_params = {
        "n_epochs": 50 
    }  
    dataloader_params = {
        "path": "data",
        "valid_size": 1000,
        "batch_size": 32,
    }  
    
    res = searcher.search(dataloader_params, trainer_params)

    with open("paramsearch_results.json", "w") as f:
        json.dump(res, f, indent=4)
