import torch
import matplotlib.pyplot as plt
from results.results import *

def plot_train_val(m_train, m_val, period, 
                    al_param=False, metric='Cross-Entropy Loss', save=True, model_name='', hybrid=0):
    """
    Plot the evolution of the 'metric' evaluated on the training and validation set during the trainining
    """
    plt.figure(figsize=(8,5))
    plt.title('Evolution of the '+metric ,fontsize=14)
    if al_param:
        al_steps = torch.Tensor(  range( 1, int(len(m_train)*period/al_param +1) )  ) *al_param
        for al_step in al_steps:
            plt.axvline(al_step, color='black')
    plt.plot(torch.Tensor(range(1,len(m_train)+1))*period, m_train, 
                color='c', marker='o', ls=':', label=metric+' train')
    plt.plot(torch.Tensor(range(1,len(m_val)+1))*period, m_val, 
                color='m', marker='o', ls=':', label=metric+' val')
    plt.axhline(min(m_val), ls=':',color='black')
    if hybrid: plt.axvline(hybrid+0.5, color='black')
    plt.xlabel('Number of Epochs')
    plt.ylabel(metric)
    plt.legend(loc = 'upper right')
    if save==True:
        plt.savefig('plots/'+model_name+' '+metric)
    plt.show()

def plot_grads_sp(first_layer, last_layer, experiment_name='', hybrid=0, save=True):
    """
    Plot the evolution of the spectral norm of the gradient of the loss 
    with respect to the weights of a certain layer, evaluated on the training 
    and validation set during the trainining
    """
    plt.figure(figsize=(8,5))
    plt.title('Evolution of the Spectral norm of the gradient of the Loss',fontsize=14)
    plt.plot(torch.Tensor(range(1,len(first_layer)+1)), first_layer, 
                color='c', marker='o', ls=':', label='First Convolution')
    plt.plot(torch.Tensor(range(1,len(last_layer)+1)), last_layer, 
                color='m', marker='o', ls=':', label='Last Linear Layer')
    if hybrid: plt.axvline(hybrid+0.5, color='black')
    plt.yscale('log')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Spectral Norm of the gradient')
    plt.legend(loc = 'upper right')
    if save==True:
        plt.savefig('plots/'+experiment_name+' gradients decay')
    plt.show()

def plot_all_opt(optimizers,plot='val_losses'):
    """
    For all the optimizers considerd plot the evolution of a 'metric' 
    evaluated on the validation set during the trainining
    """
    plt.figure(figsize=(8,5),dpi=120)
    for opt in optimizers:
        results = load_obj(opt)
        plt.plot(results[plot], label = results['optimizer_name'])
    plt.legend()
    if plot == 'val_losses':
      plt.title("Evolution of the Cross-Entropy Loss on the validation set")
      plt.ylabel("Cross-Entropu Loss")
    elif plot == 'valid_acc':
      plt.title("Evolution of the accuracy on the validation set")
      plt.ylabel("Accuracy")
    elif plot == 'grads_sn_fl':
      plt.title("Evolution of the spectral norm of the first convolution layer")
      plt.ylabel("Spectral norm")
    elif plot == 'grads_sn_ll':
      plt.title("Evolution of the spectral norm of the last linear layer")
      plt.ylabel("Spectral norm")  
    plt.savefig('plots/'+plot+' all_optimizers')
    plt.xlabel("Number of Epochs")


    plt.show()