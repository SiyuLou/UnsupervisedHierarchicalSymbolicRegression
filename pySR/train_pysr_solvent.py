import sys
sys.path.append('../')

import os
import numpy as np
import pysr
from pysr import PySRRegressor
import argparse

from data import get_data
from utils import evaluate,plot_true_vs_pred , setup_seed

def train(X, y):

    model = PySRRegressor(
        procs =4,
        populations = 8,
        # ^ 2 populations per core, so one is always running.
        population_size=50,
        # ^ Slightly larger populations, for greater diversity
        ncyclesperiteration = 50,
        # ^ Generations between migrations

        niterations=100,  # < Run forever
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-6 && complexity < 10"
            # Stop early if we find a good and simple equation
        ),
        timeout_in_seconds=60 * 60 * 1,
        # ^ Alternatively, stop after 24 hours have passed.
        maxsize = 50,
        # ^ Allow greater complexity.
        maxdepth = 10,
        # ^ But, avoid deep nesting
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["square", 
        ],
        constraints = {"/": (-1,9),
                       "square": 0,
        },
        # ^ Limit the complexity wthin each argument.
        # "inv": (-1, 9) states that the numerator has no constraint
        # but the denominator has a max complexity of 9.
        # "exp": 9 simply states that `exp` can only have 
        # an expression of complexity 9 as input
        nested_constraints={
            'square':{'square':0},
        },
        # ^ Nesting constraints on operators. For example,
        # "square(exp(x)" is not allowed, since "square": {"exp": 0}.
        complexity_of_operators={"/":0},
        # ^ Custom complexity of particular operators.
        complexity_of_constants=2,
        # ^ Punish constants more than variables
        #select_k_features=13,
        # ^ Train on only the 4 most import features
        progress=True,
        # ^ Can set to false if printing to a file.
        weight_randomize=0.1,
        # ^ Randomize the tree much more frequently
        cluster_manager=None,
        # ^ Can be set to, e.g., "slurm", to run a slurm
        # cluster. Just launch one script from the head node.
        precision = 64,
        # ^ Higher precision calculations.
        warm_start = True,
        # ^ Start from where left off. 
        turbo = True,
        # ^ Faster evaluation (exprimental)
        julia_project=None,
        # ^ Can set to the path of a folder containing the
        # "SymbolicRegression.jl" repo, for custom modifications.
        update= False,
        model_selection='accuracy',
        # ^ Don't update Julia packages
        #extra_sympy_mappings={"cos2": lambda x: np.cos(x)**2},
        # ^ Define operator for SymPy as well
        loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)
    model.fit(X, y)

    return model

    
def test(X, y, model, output_path):
    y_pred = model.predict(X)


    MSE, RMSE, MAE, R_square = evaluate(y, y_pred)
    print('RMSE: %.4f, R_square: %.4f'%(RMSE, R_square))
    plot_true_vs_pred(y, y_pred,"SymbolicRegression",  output_path, R_square, RMSE)

if __name__ =="__main__":
    pysr.julia_helpers.init_julia()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output_path', type=str, default='../result/42/01-22/Rf_final/')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    setup_seed(args.seed)
    
    exp_id = args.split_method
        
    save_folder = os.path.join(args.output_path, 'solvent_polar')
    X_train = np.load(os.path.join(save_folder,'X_train.npy'))
    y_train = np.load(os.path.join(save_folder,'y_train.npy')).reshape(-1)
    X_val = np.load(os.path.join(save_folder,'X_val.npy'))
    y_val = np.load(os.path.join(save_folder,'y_val.npy')).reshape(-1)
    X_test = np.load(os.path.join(save_folder,'X_test.npy'))
    y_test = np.load(os.path.join(save_folder,'y_test.npy')).reshape(-1)

    save_folder = os.path.join(args.output_path, 'solvent_polar', 'pysr')
    os.makedirs(save_folder, exist_ok=True)
    X_train = np.concatenate((X_train, X_val),axis=0)
    y_train = np.concatenate((y_train, y_val))
    model = train(X_train, y_train)
    test(X_test, y_test, model,save_folder)

