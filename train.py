from functools import reduce
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
from torchvision.transforms import Normalize
from torchvision.models import resnet18
import torch_optimizer as optim
import time
from plot import plot_train_val
from sklearn.model_selection import ParameterGrid

def parameter_grid(lr_min=-4, lr_max = 0, momentums_min = 0.9, momentums_max=1.0, momentums_step = 0.01):
    """
    Create hyperparameter grid, which needs to be searched
    """
    optimizers = ["SGD", "SGD+momentum", "adam", "adahessian"]
    learning_rates = 10 ** torch.arange(start=lr_min,end=lr_max+1,dtype = torch.float64)
    momentums = torch.arange(start=momentums_min,end=momentums_max,step = momentums_step, dtype = torch.float64)
    model_names = ['resnet18']
    batch_size= [100]
    nb_epochs = [5]
    reducers = [100]
 
    hyperparameters = {
        "optimizers": optimizers,
        "learning_rates":learning_rates,
        "momentums" : momentums,
        "model_names":model_names,
        "batch_size" : batch_size,
        "nb_epochs":nb_epochs,
        "reducers":reducers
    }
    
    return ParameterGrid(hyperparameters)

def parameter_grid_search(plot = True, print_ = True):
    """
    Searches the hyperparameter grid
    """
    PG = parameter_grid()
    curr_test_acc = 0.0
    best_return, best_hyperparameters = {}, {}
    iteration_number, total_parameter_grid_points = 1, len(PG)
    for hyperparameters in PG:
        optimizer = hyperparameters["optimizers"]
        lr = hyperparameters["learning_rates"]
        momentum = hyperparameters["momentums"]
        model_name = hyperparameters["model_names"]
        batch_size = hyperparameters["batch_size"]
        nb_epochs = hyperparameters["nb_epochs"]
        reduce = hyperparameters["reducers"]
        returns = run_experiment(optimizer_name=optimizer, model_name=model_name, nb_epochs = nb_epochs, batch_size = batch_size, plot=plot, reduce=reduce,print_  = print_,lr = lr, momentum = momentum)
        if curr_test_acc < returns["test_acc"]:
            best_return = returns.copy()
            best_hyperparameters = hyperparameters.copy()
        print(f"\n---------- Finished {iteration_number}/{total_parameter_grid_points} ----------------------------\n")
        iteration_number +=1
    return best_return, best_hyperparameters



def run_experiment(optimizer_name="optimizer", model_name='resnet18', nb_epochs = 10, 
                            batch_size = 100, plot=True, reduce=100, print_  = True,lr = 0.005, momentum = 0.9):
    '''
    Run Experiment
    '''
    experiment_name = model_name + '_' + optimizer_name
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    if print_:
        print("Device used: ", device,'\n')

    # loading the data
    if print_:
        print("Dataset: MNIST")
    train_ds = MNIST('./data/' +"mnist", train=True, transform=ToTensor(),download=True)
    test_ds = MNIST('./data/' +"mnist", train=False, transform=ToTensor(),download=True)
    
    # reduce the dataset dimension by 10 times
    if reduce!=None:
        train_filter = list(range(0, len(train_ds), reduce))
        train_ds = Subset(train_ds, train_filter)
        test_filter = list(range(1, len(test_ds), reduce))
        test_ds = Subset(test_ds, test_filter)
    
    if print_:
        print("Training size: ", len(train_ds))
        print("Test size: ", len(test_ds))
        print("Dimension Images: 28x28")
        print('Number of classes: 10 \n')    

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # model
    if print_:
        print('Model: ', model_name.capitalize())
    if model_name=='resnet18':
        model = resnet18(num_classes=10, pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        raise ValueError('The model selected doesn\'t exists or it is not already implemented')
    model = model.to(device) 
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_:
        print('Number of parameters: ',n_parameters,'\n')
       
    # training
    if print_:
        print('Loss Function: Cross Entropy Loss')
    criterion = torch.nn.CrossEntropyLoss()
    if print_:
        print('Optimizer: ', optimizer_name.capitalize())
    if optimizer_name=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) # lr = 5*1e-3
    elif optimizer_name=='SGD+momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum) # lr = 5*1e-3 momentum = 9* 1e-1
    elif optimizer_name=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)# lr = 5*1e-3
    elif optimizer_name=='adahessian':
        optimizer = optim.Adahessian(model.parameters(),
                                    lr= lr, # lr = 1.0
                                    betas= (0.9, 0.999),
                                    eps= 0.0001,
                                    weight_decay=0.0,
                                    hessian_power=1.0,
                                )
    else:
        raise ValueError('The optimizer selected doesn\'t exists or it is not already implemented')
    
    # Train the model and measure total training time
    start = time.time()
    train_losses, val_losses,spectral_norms_last_layer = train(model, train_dl, test_dl, optimizer,criterion, device, experiment_name, nb_epochs)
    end = time.time()
    total_training_time = end-start

    if print_:
        print('Training time: {0:.3f} seconds'.format(total_training_time))

    # load model
    path = "./model_weights/" + experiment_name + ".pth"
    model.load_state_dict(torch.load(path))

    # Accuracy Train
    train_acc = test(model,train_dl, device)
    if print_:
        print('\n\nAccuracy Train: {}%'.format(train_acc))
    
    # Accuracy Test
    test_acc = test(model,test_dl, device)
    if print_:    
        print('Accuracy Test: {}%'.format(test_acc))

    # plot evolution losses
    if plot==True: plot_train_val(train_losses, val_losses, period=1, model_name=experiment_name)

    # return time, losses and accuracies
    returns = {
        "training_time" : total_training_time,
        "train_losses":train_losses,
        "val_losses":val_losses,
        "train_acc":train_acc,
        "test_acc":test_acc,
        "spectral_norms_last_layer":spectral_norms_last_layer
    }
    return returns


def test(model, dataloader, device):
    '''
    Test a model returning the Accuracy (in percentage)
    '''
    correct = 0
    total = 0

    with torch.no_grad():
        for data in dataloader:
            images, labels =data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return (100 * correct / total)

  

def train(model, train_loader, val_loader, optimizer, criterion, device, model_name, nb_epochs = 10):
    """
    Train a model
    """
    train_losses = []
    val_losses = []
    spectral_norms_last_layer = []
    for epoch in range(nb_epochs):
        train_loss = 0
        model.train()
        i = 0
        spectral_norm = 0
        ##### Training ######
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward(create_graph = True,)
            # Update the Gradient
            optimizer.step()
            # Collect the Losses
            train_loss += loss.data.item()

            i += 1
            spectral_norm += torch.linalg.matrix_norm(model.fc.weight.grad, ord = 2)

        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss) 
        #
        spectral_norms_last_layer.append(spectral_norm)
        print(f"Epoch {epoch} / {nb_epochs} average spectral norm of last layer for single batch {spectral_norm/i}\n")
        #
        ##### Evaluation #####
        model.eval()
        val_loss = 0
        for data in val_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                val_preds = model(inputs)
                val_loss += criterion(val_preds, targets).data.item()
        val_loss = val_loss / len(val_loader) 
        val_losses.append(val_loss)
        # save best model in validation
        if val_loss <= min(val_losses):
            torch.save(model.state_dict(), "./model_weights/" + model_name + ".pth")
        print("Epoch", epoch+1, "/", nb_epochs, "train loss:", train_loss, "valid loss:", val_loss)
    
    return train_losses, val_losses, spectral_norms_last_layer