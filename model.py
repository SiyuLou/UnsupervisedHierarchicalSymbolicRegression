import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalNeuralNetwork(nn.Module):
    '''
    Construct Hierarchical neural network to extract latent variables
    '''
    def __init__(self, 
                 input_neurons,
                 hidden_neuron = 50, 
                 out_neuron = 1,
                 num_submodels = 2,
                 activation = 'sigmoid',
        ):
        super(HierarchicalNeuralNetwork, self).__init__()
        
        self.num_submodels = num_submodels
        submodels = []
        for i in range(num_submodels):
            submodels.append(nn.Sequential(
                nn.Linear(input_neurons[i], hidden_neuron),
                nn.LeakyReLU(),
                nn.Linear(hidden_neuron, hidden_neuron),
                nn.LeakyReLU(),
                nn.Linear(hidden_neuron, 1),
                )
            )

        self.submodels = nn.ModuleList(submodels)


        self.output_layer = nn.Sequential(
                nn.Linear(num_submodels, hidden_neuron),
                nn.LeakyReLU(),
                nn.Linear(hidden_neuron, hidden_neuron),
                nn.LeakyReLU(),
                nn.Linear(hidden_neuron, out_neuron)
        )
        
        self.activation = activation,
 
    def forward(self, x):
        func_outputs = []
        
        for i in range(self.num_submodels):
            input_ = x[i]
            output_ = self.submodels[i](input_)
            func_outputs.append(output_)

        func_outputs = torch.cat(func_outputs, dim=1)
        output = self.output_layer(func_outputs)
        if self.activation=='sigmoid':
            output = torch.sigmoid(output)
        return output

    def _get_hidden_layer(self, x):
        func_inputs = []
        func_outputs = []
        
        for i in range(self.num_submodels):
            input_ = x[i]
            output_ = self.submodels[i](input_)

            func_inputs.append(input_.detach())
            func_outputs.append(output_.detach())
        
        func_outputs = torch.cat(func_outputs, dim=1)
        return func_inputs, func_outputs


