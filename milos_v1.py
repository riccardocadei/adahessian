from ray import tune
import torch
def model_tunning():
    optimizers_ = ["SGD", "SGD+momentum", "adam", "adahessian"]
    lr_min, lr_max = 1e-4, 1e-1
    momentums_min, momentums_max, momentums_step = 0.9, 1.0, 0.01
    momentums_ = torch.range(start=momentums_min,end=momentums_max,step=momentums_step).tolist()
    models = ['resnet18']
    batch_size= []
    num_epochs = []
    reducers = []


    config_ = {
        "optimizer_": tune.choice(optimizers_),
        "lr": tune.loguniform(lr_min, lr_max),
        "momentum" : tune.grid_search(momentums_),
    }
    results=[]

    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }