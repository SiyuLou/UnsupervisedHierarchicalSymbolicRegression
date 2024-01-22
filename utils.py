import random
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
def evaluate(y_test, y_pred):
    MSE = np.sum(np.abs(y_test - y_pred)**2) /y_test.shape[0]
    RMSE=np.sqrt(MSE)
    MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
    R_square=1-(((y_test-y_pred)**2).sum()/((y_test-y_test.mean())**2).sum())
    R_square_2 = r2_score(y_test, y_pred)
    return MSE, RMSE, MAE, R_square


def plot_true_vs_pred(y_test, y_pred, model_name, savefolder, r2_score, rmse_score, set_ylim=True):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(1, 1, figsize=(6, 6), dpi=300)

    # Create a diagonal line for reference
    max_value = np.max(y_test)
    min_value = np.min(y_test)
    plt.plot(np.linspace(min_value, max_value, 100), np.linspace(min_value, max_value, 100),
             linewidth=1, linestyle='--', color='black')

    plt.scatter(y_test, y_pred, c='#00008B', s=15, alpha=0.4)
    plt.xlabel('True value', fontproperties='Serif', size=20)
    plt.ylabel("Predict value", fontproperties='Serif', size=20)
    #plt.title(f"{model_name}", fontproperties='Arial', size=30)

    axes.xaxis.set_major_locator(plt.MaxNLocator(5))
    axes.yaxis.set_major_locator(plt.MaxNLocator(5))
    x_min = y_test.min()
    y_max = y_pred.max()
    delta = (y_pred.max() - y_pred.min())/5
    plt.text(x_min, y_max-delta/2, f'R² = {r2_score:.3f}', fontsize=30, fontproperties='Serif')
    plt.text(x_min, y_max-delta, f'RMSE = {rmse_score:.3f}', fontsize=30, fontproperties='Serif')
    if set_ylim:
        plt.ylim([0, 1.1])
        plt.xlim([0, 1.1])

    plt.tight_layout()
    plt.savefig(os.path.join(savefolder, f'True_vs_Predicted_{model_name}.svg'), bbox_inches='tight', dpi=300)
    plt.show()
    plt.close('all')

def setup_seed(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


def adjust_learning_rate(lr, optimizer, epoch):
    if epoch < 200:
        lr = lr
    else:
        lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




def test(model,
         device,
         test_loader,
         ):

    pred_all = []
    label_all = []
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data1,data2, label = data[0].to(device),data[1].to(device), label.to(device)
            pred = model(data1,data2)
            pred_all.append(pred)
            label_all.append(label)
    pred_all = torch.cat(pred_all)
    label_all = torch.cat(label_all)

    return pred_all.squeeze().detach().cpu().numpy(), \
            label_all.detach().cpu().numpy()

def get_hidden_variables(dataloader, model, device, num_submodels):
    func_inputs_sequences = []
    func_outputs_sequences = []

    for i in range(num_submodels):
        func_inputs_sequences.append([])
        func_outputs_sequences.append([])
    
    func_inputs_sequence_final = []
    func_outputs_sequence_final = []
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            num_x = len(data)
            x = []
            for i in range(num_x):
                x.append(data[i].to(device))
            label = label.to(device)
            x = tuple(x)
            func_inputs, func_outputs = model._get_hidden_layer(x)
            for j in range(num_submodels):
                func_inputs_sequences[j].append(func_inputs[j])
                func_outputs_sequences[j].append(func_outputs[:,j].unsqueeze(1))
            
            func_inputs_sequence_final.append(func_outputs)
            func_outputs_sequence_final.append(label.unsqueeze(1))
    for i in range(num_submodels):
        func_inputs_sequences[i] = torch.cat(func_inputs_sequences[i]).detach().cpu().numpy()
        func_outputs_sequences[i] = torch.cat(func_outputs_sequences[i]).detach().cpu().numpy()
    
    func_inputs_sequence_final = torch.cat(func_inputs_sequence_final).detach().cpu().numpy()
    func_outputs_sequence_final = torch.cat(func_outputs_sequence_final).detach().cpu().numpy()
    
    func_inputs_sequences.append(func_inputs_sequence_final)
    func_outputs_sequences.append(func_outputs_sequence_final)
    return func_inputs_sequences, func_outputs_sequences

