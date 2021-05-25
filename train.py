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


def run_experiment(optimizer_name="optimizer", model_name='resnet18', nb_epochs = 10, 
                            batch_size = 100, plot=True, reduce=100):
    '''
    Run Experiment
    '''
    experiment_name = model_name + '_' + optimizer_name
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device used: ", device,'\n')

    # loading the data
    print("Dataset: MNIST")
    train_ds = MNIST('./data/' +"mnist", train=True, transform=ToTensor(),download=True)
    test_ds = MNIST('./data/' +"mnist", train=False, transform=ToTensor(),download=True)
    
    # reduce the dataset dimension by 10 times
    if reduce!=None:
        train_filter = list(range(0, len(train_ds), reduce))
        train_ds = Subset(train_ds, train_filter)
        test_filter = list(range(1, len(test_ds), reduce))
        test_ds = Subset(test_ds, test_filter)

    print("Training size: ", len(train_ds))
    print("Test size: ", len(test_ds))
    print("Dimension Images: 28x28")
    print('Number of classes: 10 \n')    

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # model
    print('Model: ', model_name.capitalize())
    if model_name=='resnet18':
        model = resnet18(num_classes=10, pretrained=False)
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    else:
        raise ValueError('The model selected doesn\'t exists or it is not already implemented')
    model = model.to(device) 
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: ',n_parameters,'\n')
       
    # training
    print('Loss Function: Cross Entropy Loss')
    criterion = torch.nn.CrossEntropyLoss()
    print('Optimizer: ', optimizer_name.capitalize())
    if optimizer_name=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    elif optimizer_name=='SGD+momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    elif optimizer_name=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    elif optimizer_name=='adahessian':
        optimizer = optim.Adahessian(model.parameters(),
                                    lr= 1.0,
                                    betas= (0.9, 0.999),
                                    eps= 0.0001,
                                    weight_decay=0.0,
                                    hessian_power=1.0,
                                )
    else:
        raise ValueError('The optimizer selected doesn\'t exists or it is not already implemented')
    
    start = time.time()
    train_losses, val_losses = train(model, train_dl, test_dl, optimizer,criterion, device, experiment_name, nb_epochs)
    end = time.time()
    print('Training time: {0:.3f} seconds'.format(end-start))

    # load model
    path = "./model_weights/" + experiment_name + ".pth"
    model.load_state_dict(torch.load(path))

    # test
    print('\n\nAccuracy Train: {}%'.format(test(model,train_dl, device)))
    print('Accuracy Test: {}%'.format(test(model,test_dl, device)))

    # plot evolution losses
    if plot==True: plot_train_val(train_losses, val_losses, period=1, model_name=experiment_name)


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
    for epoch in range(nb_epochs):
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
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)

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
    
    return train_losses, val_losses