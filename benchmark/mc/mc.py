import numpy as np
import torch
import torch.nn as nn
import yaml
import scipy.io as sio
import pandas as pd

""" Controller network utils """

# PyTorch model for mc controller net
class MC_Control_NN(nn.Module):

    def __init__(self, layer_1_size=16, layer_2_size=16):
        super(MC_Control_NN, self).__init__()
        self.fc1 = nn.Linear(2, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Load yaml to torch model
def load_model_dict(model_dict: dict, network: nn.Module):
    def normalize_keys(d: dict) -> dict:
        """Convert keys like 'x1', 'x_1' or '1' to int."""
        new_d = {}
        for k, v in d.items():
            if isinstance(k, int):
                new_d[k] = v
            elif isinstance(k, str) and k.isdigit():
                new_d[int(k)] = v
            elif isinstance(k, str) and k.startswith('x_') and k[2:].isdigit():
                new_d[int(k[2:])] = v
            elif isinstance(k, str) and k.startswith('x') and k[1:].isdigit():
                new_d[int(k[1:])] = v
            else:
                raise ValueError(f"Unsupported key format: {k}")
        return new_d

    def to_tensor(val):
        """Convert to tensor; if float, wrap in list first."""
        if isinstance(val, float):
            return torch.FloatTensor([val])
        return torch.FloatTensor(val)

    weights = normalize_keys(model_dict['weights'])
    offsets = normalize_keys(model_dict['offsets'])

    state_dict = {
        'fc1.weight': torch.FloatTensor(weights[1]),
        'fc1.bias': to_tensor(offsets[1]),
        'fc2.weight': torch.FloatTensor(weights[2]),
        'fc2.bias': to_tensor(offsets[2]),
        'fc3.weight': torch.FloatTensor(weights[3]).reshape(1, -1),  # Ensure shape is [1, N]
        'fc3.bias': to_tensor(offsets[3])
    }

    network.load_state_dict(state_dict)

# Dump torch model to yaml
def dump_model_dict(filename, network: nn.Module):
    model_dict = {}
    model_dict['activations'] = {}
    if isinstance(network, MC_Control_NN):
        model_dict['activations'][1] = 'Sigmoid'
        model_dict['activations'][2] = 'Sigmoid'
    else:
        raise NotImplementedError
    model_dict['activations'][3] = 'Tanh'
    model_dict['weights'] = {}
    model_dict['offsets'] = {}
    for layer in [1, 2, 3]:
        model_dict['weights'][layer] = network.state_dict()[f'fc{layer}.weight'].tolist()
        model_dict['offsets'][layer] = network.state_dict()[f'fc{layer}.bias'].tolist()
    with open(filename, 'w') as f:
        yaml.dump(model_dict, f)
    return

""" MC utils """

# Simulate traj_pos and traj_vel from an initial state (replace this with actual simulator later)
def mc_simulate(pos_0, vel_0, net: MC_Control_NN, length=111, steepness=0.0025):
    net.eval()
    traj_pos = np.zeros(length)
    traj_vel = np.zeros(length)
    traj_pos[0] = pos_0
    traj_vel[0] = vel_0

    traj_nn_in_1 = []
    traj_nn_in_2 = []
    traj_nn_out = []
    
    for i in range(1, length):
        traj_pos[i] = traj_pos[i - 1] + traj_vel[i - 1]
        inputs = torch.FloatTensor([traj_pos[i - 1], traj_vel[i - 1]])
        u = net(inputs).detach().numpy()

        traj_nn_in_1.append(traj_pos[i - 1])
        traj_nn_in_2.append(traj_vel[i - 1])
        traj_nn_out.append(u)

        traj_vel[i] = traj_vel[i - 1] + 0.0015 * u - steepness * np.cos(3 * traj_pos[i - 1])
        # simulator constraints
        if (traj_vel[i] > 0.07): traj_vel[i] = 0.07
        if (traj_vel[i] < -0.07): traj_vel[i] = -0.07
        if (traj_pos[i] > 0.6): traj_pos[i] = 0.6
        if (traj_pos[i] < -1.2): traj_pos[i] = -1.2
        if (traj_pos[i] == -1.2 and traj_vel[i] < 0): traj_vel[i] = 0

    return traj_nn_in_1, traj_nn_in_2, traj_nn_out, traj_pos, traj_vel
    

# STL robustness of MC
def mc_robustness(traj_pos):
    taus = 110
    return np.max(traj_pos) - 0.45, taus
