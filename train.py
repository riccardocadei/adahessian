
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch_optimizer as optim
torch.manual_seed(0)

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.models import resnet18

from sklearn.model_selection import ParameterGrid
from results.results import *

import time
from plot import *


def parameter_grid(lr_min=0.1, lr_max = 1, step=0.2):
    """
    Create hyperparameter grid, which needs to be searched
    """
    optimizers = ["adahessian"] #[ "SGD", "SGD+momentum", "adam", "adahessian"]
    learning_rates = torch.arange(start=lr_min,end=lr_max,step=step,dtype = torch.float64)
    momentums = [0.9]
    model_names = ['resnet18']
    batch_size= [100]
    nb_epochs = [15]
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

def parameter_grid_search(plot = False, print_ = False):
    """
    Hyper-parameters tuning using Grid Search
    """
    PG = parameter_grid()
    curr_test_acc = 0.0
    best_return, best_hyperparameters = {}, {}
    for iteration_number, hyperparameters in enumerate(PG):
        optimizer = hyperparameters["optimizers"]
        lr = hyperparameters["learning_rates"]
        momentum = hyperparameters["momentums"]
        model_name = hyperparameters["model_names"]
        batch_size = hyperparameters["batch_size"]
        nb_epochs = hyperparameters["nb_epochs"]
        reduce = hyperparameters["reducers"]
        print(f"\n---------- Experiment {iteration_number+1}/{len(PG)} ----------\n")
        print(f"Method: {optimizer}")
        print(f"Learning Rate: {lr}")
        valid_accs = []
        n_test = 5
        for _ in range(n_test):
            returns = run_experiment(optimizer_name=optimizer, model_name=model_name, nb_epochs = nb_epochs, batch_size = batch_size, plot=plot, reduce=reduce,print_  = print_,lr = lr, momentum = momentum)
            valid_accs.append(returns["valid_acc"])
        valid_acc = torch.Tensor(valid_accs).mean().item()
        valid_acc_std = torch.Tensor(valid_accs).std().item()
        if curr_test_acc < valid_acc:
            best_return = returns.copy()
            best_hyperparameters = hyperparameters.copy()
        print(f"Accuracy Validation: {valid_acc} ± {valid_acc_std}")
    return best_return, best_hyperparameters

def run_experiment(optimizer_name="optimizer", model_name='resnet18', nb_epochs = 15, 
                            batch_size = 100, plot=True, reduce=100, print_= True, lr = 0.005, momentum = 0.9):
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
        valid_filter = list(range(1, len(test_ds), reduce))
        valid_ds = Subset(test_ds, valid_filter)
        test_filter = list(range(2, len(test_ds), reduce))
        test_ds = Subset(test_ds, test_filter)
    
    if print_:
        print("Training size: ", len(train_ds))
        print("Validation size: ", len(valid_ds))
        print("Test size: ", len(test_ds))
        print("Dimension Images: 28x28")
        print('Number of classes: 10 \n')    

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)
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
    hybrid = False
    if print_:
        print('Optimizer: ', optimizer_name.capitalize())
        print('Learning Rate: ', lr,'\n')
    if optimizer_name=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) # lr = 5*1e-3
    elif optimizer_name=='SGD+momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum) # lr = 5*1e-3 momentum = 0.9
    elif optimizer_name=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) # lr = 5*1e-3
    elif optimizer_name=='adahessian':
        optimizer = optim.Adahessian(model.parameters(),
                                    lr= lr, # lr = 1
                                    betas= (0.9, 0.999),
                                    eps= 0.0001,
                                    weight_decay=0.0,
                                    hessian_power=1.0,
                                )
    elif optimizer_name=='hybrid':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum) # lr = 5*1e-3 momentum = 0.9
        hybrid = True
    else:
        raise ValueError('The optimizer selected doesn\'t exists or it is not already implemented')
    
    # Train the model and measure total training time
    start = time.time()
    train_losses, valid_losses,grads_sn_fl, grads_sn_ll, change = train(model, train_dl, test_dl, optimizer,criterion, device, experiment_name, nb_epochs, hybrid=hybrid, calculate_spectral_norms = True, print_=print_)
    end = time.time()
    total_training_time = end-start

    if print_:
        print('Training time: {0:.3f} seconds'.format(total_training_time))

    # load model
    path = "./model_weights/" + experiment_name + ".pth"
    model.load_state_dict(torch.load(path))

    # Accuracy
    train_acc = test(model,train_dl, device)
    valid_acc = test(model,valid_dl, device)
    test_acc = test(model,test_dl, device)
    if print_:
        print('\n\nAccuracy Train: {}%'.format(train_acc))   
        print('Accuracy Validation: {}%'.format(valid_acc))   
        print('Accuracy Test: {}%\n'.format(test_acc))

    # plot evolution losses
    if plot: 
        plot_train_val(train_losses, valid_losses, period=1, model_name=experiment_name, hybrid=change)
        plot_grads_sp(grads_sn_fl, grads_sn_ll, experiment_name=experiment_name, hybrid=change)

    # return time, losses and accuracies
    returns = {
        "optimizer_name" : optimizer_name,
        "training_time" : total_training_time,
        "train_losses": train_losses,
        "val_losses":valid_losses,
        "train_acc":train_acc,
        "valid_acc":valid_acc,
        "test_acc":test_acc,
        "grads_sn_fl":grads_sn_fl,
        "grads_sn_ll":grads_sn_ll
    }

    save_obj(returns,optimizer_name)

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

def train(model, train_loader, valid_loader, optimizer, criterion, device, model_name, nb_epochs = 10, hybrid=False, print_=True, calculate_spectral_norms = True):
    """
    Train a model
    """
    train_losses = []
    valid_losses = []
    grads_sn_fl = []
    grads_sn_ll = []
    change = 0
    for epoch in range(nb_epochs):
        grad_sn_fl = 0
        grad_sn_ll = 0
        train_loss = 0
        model.train()
        ##### Training ######
        for data in train_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward(create_graph = True)
            # Update the Gradient
            optimizer.step()
            # Collect the Losses
            train_loss += loss.data.item()
            # Save the spectral norm of the gradient
            if calculate_spectral_norms:
                grad_sn_fl += torch.linalg.matrix_norm(model.conv1.weight.grad, ord = 2).sum()
                grad_sn_ll += torch.linalg.matrix_norm(model.fc.weight.grad, ord = 2)
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        if calculate_spectral_norms:
            grad_sn_fl = grad_sn_fl / (len(train_loader)*64)
            grad_sn_ll = grad_sn_ll / (len(train_loader)*64)
            grads_sn_fl.append(grad_sn_fl)
            grads_sn_ll.append(grad_sn_ll)


        ##### Evaluation #####
        model.eval()
        valid_loss = 0
        for data in valid_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                valid_preds = model(inputs)
                valid_loss += criterion(valid_preds, targets).data.item()
        valid_loss = valid_loss / len(valid_loader) 
        valid_losses.append(valid_loss)
        # save best model in validation
        if valid_loss <= min(valid_losses):
            torch.save(model.state_dict(), "./model_weights/" + model_name + ".pth")
        
        if hybrid and valid_loss < 1 and change==0:
            optimizer = optim.Adahessian(model.parameters(),
                                    lr= 0.001, # lr = 1
                                    betas= (0.9, 0.999),
                                    eps= 0.0001,
                                    weight_decay=0.0,
                                    hessian_power=1.0,
                                )
            change = epoch+1
        if print_: print("Epoch", epoch+1, "/", nb_epochs, "train loss:", train_loss, "valid loss:", valid_loss)

    return train_losses, valid_losses, grads_sn_fl, grads_sn_ll, change
