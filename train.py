import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm, trange
from datetime import date
today = date.today()
import numpy as np

from utils import evaluate, plot_true_vs_pred, setup_seed, adjust_learning_rate,get_hidden_variables
from data import get_data, get_data_submodel
from model import HierarchicalNeuralNetwork 
from config import feature_names, feature_nums, feature_names_solute, feature_nums_solute,feature_names_functional,feature_nums_functional

def train(model,
          feature_nums,
          device,
          train_loader,
          optimizer,
          loss_fn,
    ):
    
    running_loss = 0.
    model.train()
    pred_all = []
    label_all = []

    for i, (data, label) in enumerate(train_loader):
        num_x = len(data)
        x = []
        for i in range(num_x):
            x.append(data[i].to(device))
        label = label.to(device)
        x = tuple(x)
        optimizer.zero_grad()
        
        pred = model(x)
        loss = loss_fn(pred, label)
        loss.backward()

        optimizer.step()
        pred_all.append(pred)
        label_all.append(label)
        running_loss += loss.item()
    
    avg_loss = running_loss / (i+1)
    pred_all = torch.cat(pred_all)
    label_all = torch.cat(label_all)
    return avg_loss, pred_all.squeeze().detach().cpu().numpy(), \
            label_all.detach().cpu().numpy()


def test(model,
         feature_nums,
         device,
         test_loader,
         loss_fn
         ):

    running_vloss = 0.
    pred_all = []
    label_all = []
    model.eval()
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            num_x = len(data)
            x = []
            for i in range(num_x):
                x.append(data[i].to(device))
            label = label.to(device)
            x = tuple(x)
            pred = model(x)
            loss = loss_fn(pred, label)
            running_vloss += loss.item()
            pred_all.append(pred)
            label_all.append(label)
    avg_loss = running_vloss / (i+1)
    pred_all = torch.cat(pred_all)
    label_all = torch.cat(label_all)

    return avg_loss, pred_all.squeeze().detach().cpu().numpy(), \
            label_all.squeeze().detach().cpu().numpy()

def model_train_test(args, 
                     feature_names, 
                     output_path, 
                     feature_nums,
                     num_submodels,
                     save_dir_list,
                     train_loader,
                     val_loader,
                     test_loader,
                     activation,
                     set_ylim = True):
    '''
    Input:
        feature_names, List[Str]: name of the features
        putput_path, Str: output path for the file
        feature_nums, List[Int]:  number of features in each category


    Output:
        RMSE and R_square on test data

    '''
    # determin te device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f'running experiment on device {device}')

    # Seperate data accurding to category, e.g. Solvent related features and 
    # Solution related features
 
    model = HierarchicalNeuralNetwork(input_neurons = feature_nums,
                                      hidden_neuron=args.hidden_dim,
                                      num_submodels = num_submodels,
                                      activation = activation).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = nn.MSELoss()
    train_losses = []
    test_losses = []
    pbar=trange(args.epochs, desc="training DNN", unit="epoch")
    best_MSE = np.inf
    patience = 0
    for epoch in pbar:
        adjust_learning_rate(args.lr, optimizer, epoch)
        train_loss, train_preds, train_labels = train(model, feature_nums,  device, train_loader, optimizer, loss_fn)
        val_loss, val_preds, val_labels = test(model, feature_nums, device, val_loader, loss_fn)
        train_losses.append(train_loss)
        test_losses.append(val_loss)
        MSE, RMSE, MAE, R_square = evaluate(val_labels, val_preds)
        pbar.set_description('Epoch %d: train loss: %.4f, val loss: %.4f RMSE: %.4f, R_square: %.4f'%(epoch,train_loss, val_loss, RMSE, R_square))
        if MSE < best_MSE:
            best_MSE = MSE
            torch.save({"epoch":epoch,
                        "model":model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "train_losses":train_losses,
                        "test_losses":test_losses
                        },
                        os.path.join(output_path,f"model_best.pth.tar")
            )
            patience = 0
           #print("save model successfully !\n")
        patience += 1
    ## load best model
    checkpoint = torch.load(os.path.join(output_path, f"model_best.pth.tar"))["model"]
    model.load_state_dict(checkpoint)
    model.to(device)
    test_loss, test_preds, test_labels = test(model, feature_nums, device, test_loader, loss_fn)

    MSE, RMSE, MAE, R_square = evaluate(test_labels, test_preds)
    plot_true_vs_pred(test_labels, test_preds,"HierarchyDNN",  output_path, R_square, RMSE, set_ylim=set_ylim)
    

    func_inputs_train, func_outputs_train = get_hidden_variables(train_loader,model, device, num_submodels)
    func_inputs_val, func_outputs_val = get_hidden_variables(val_loader, model, device, num_submodels)
    func_inputs_test, func_outputs_test = get_hidden_variables(test_loader, model, device, num_submodels)
    for i, save_dir in enumerate(save_dir_list):   
        save_folder = os.path.join(output_path, save_dir_list[i])
        os.makedirs(save_folder, exist_ok=True)
        np.save(os.path.join(save_folder,'X_train.npy'),func_inputs_train[i])
        np.save(os.path.join(save_folder,'y_train.npy'),func_outputs_train[i])
        np.save(os.path.join(save_folder,'X_val.npy'),func_inputs_val[i])
        np.save(os.path.join(save_folder,'y_val.npy'),func_outputs_val[i])
        np.save(os.path.join(save_folder,'X_test.npy'),func_inputs_test[i])
        np.save(os.path.join(save_folder,'y_test.npy'),func_outputs_test[i])

    return R_square, MSE

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', type=str, default='./result/')
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-2) 
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--xlsx_file", type =str, default='data/TLC_data.xlsx')
    parser.add_argument("--hidden_dim", type = int, default=50)
    args = parser.parse_args()
    setup_seed(args.seed)
    
    exp_id = 'Rf_final'
    num_submodels = 2

    args.output_path = os.path.join(args.output_path, str(args.seed), f'{today.strftime("%m-%d")}')
    save_folder = os.path.join(args.output_path, exp_id)
    os.makedirs(save_folder, exist_ok=True)
    ## Rf_governing_equation
    file = open(os.path.join(save_folder, 'result.txt'), 'w')
    file.write(f"input feature dimension {len(feature_names)}: {feature_names} \n") 
    save_dir_list = ['solvent_polar','solute_polar','final']
    # load data
    train_loader, val_loader, test_loader = get_data(feature_names, 
                                                     feature_nums,
                                                     xlsx_file=args.xlsx_file,
                                                     random_state=args.seed,
                                                     savepath=save_folder)

    R_square, MSE = model_train_test(args, 
                                     feature_names=feature_names,
                                     output_path=save_folder, 
                                     feature_nums=feature_nums,
                                     num_submodels = num_submodels,
                                     save_dir_list = save_dir_list,
                                     train_loader= train_loader,
                                     val_loader = val_loader,
                                     test_loader = test_loader,
                                     activation = 'sigmoid'
                                     )
    file.write(f"R_square: {R_square}; MSE: {MSE} \n\n")
    print(f"R_square: {R_square}; MSE: {MSE} \n\n")
    file.close()
    
    #### xi governing equation
    exp_id = 'solute_polarity_index'
    num_submodels = 2
    data_save_folder = os.path.join(save_folder, 'solute_polar')
    save_folder = os.path.join(args.output_path, exp_id)
    os.makedirs(save_folder, exist_ok=True)
    file = open(os.path.join(save_folder, 'result.txt'), 'w')
    file.write(f"input feature dimension {len(feature_names_solute)}: {feature_names_solute} \n") 
    save_dir_list = ['FG_distribution_polarity','FG_polarity','final']
    # load data
    train_loader, val_loader, test_loader = get_data_submodel(save_folder = data_save_folder, 
                                                              feature_nums = feature_nums_solute
                                                              )

    R_square, MSE = model_train_test(args, 
                                     feature_names=feature_names_solute,
                                     output_path=save_folder, 
                                     feature_nums=feature_nums_solute,
                                     num_submodels = num_submodels,
                                     save_dir_list = save_dir_list,
                                     train_loader= train_loader,
                                     val_loader = val_loader,
                                     test_loader = test_loader,
                                     activation = None,
                                     set_ylim = False
                                     )
    file.write(f"R_square: {R_square}; MSE: {MSE} \n\n")
    print(f"R_square: {R_square}; MSE: {MSE} \n\n")
    file.close()
    #### beta governing equation
    exp_id = 'FG_polarity_index'
    num_submodels = 5
    data_save_folder = os.path.join(save_folder, 'FG_polarity')
    save_folder = os.path.join(args.output_path, exp_id)
    os.makedirs(save_folder, exist_ok=True)
    file = open(os.path.join(save_folder, 'result.txt'), 'w')
    file.write(f"input feature dimension {len(feature_names_functional)}: {feature_names_functional} \n") 
    save_dir_list = ['FG_1','FG_2','FG_3','FG_4','FG_5','final']
    # load data
    train_loader, val_loader, test_loader = get_data_submodel(save_folder = data_save_folder, 
                                                              feature_nums = feature_nums_functional
                                                              )

    R_square, MSE = model_train_test(args, 
                                     feature_names=feature_names_functional,
                                     output_path=save_folder, 
                                     feature_nums=feature_nums_functional,
                                     num_submodels = num_submodels,
                                     save_dir_list = save_dir_list,
                                     train_loader= train_loader,
                                     val_loader = val_loader,
                                     test_loader = test_loader,
                                     activation = None,
                                     set_ylim = False
                                     )
    file.write(f"R_square: {R_square}; MSE: {MSE} \n\n")
    print(f"R_square: {R_square}; MSE: {MSE} \n\n")
    file.close()

  
