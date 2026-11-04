import os
import numpy as np
import json
import yaml
import argparse
from mc import MC_Control_NN, load_model_dict, mc_simulate, mc_robustness
from scipy.io import savemat

# Parse command-line arguments for UUV simulation settings
parser = argparse.ArgumentParser()
parser.add_argument('--init_pos', type=float, default=-0.5, help='Initial position')
parser.add_argument('--init_vel', type=float, default=0.0, help='Initial velocity')
parser.add_argument('--nn_path', type=str, default='mc_sig_2x16_broken.yml', help='Path to the YAML file of NN controller')
args = parser.parse_args()

# Step 1: Load the pre-trained controller model
# Always resolve to absolute path (relative to current working directory)
nn_path = os.path.abspath(args.nn_path)

# Load from YAML file
with open(nn_path, 'rb') as f:
    model_dict = yaml.safe_load(f)

# Create model instance
net = MC_Control_NN()

# Load parameters into the model
load_model_dict(model_dict, net)

# Step 2: Simulate the system using the controller
# For Mountain Car:
init_pos = args.init_pos  # Initial position from command line
init_vel = args.init_vel  # Initial velocity from command line
traj_nn_in_1, traj_nn_in_2, traj_nn_out, traj_pos, traj_vel = mc_simulate(pos_0=init_pos, vel_0=init_vel, net=net)



# Step 3: Evaluate the STL specification robustness of the trajectory
# For Mountain Car:
rob, taus = mc_robustness(traj_pos)

# for FL
diagInfo = {
    'rob': float(rob),  # 确保是 float 类型
    'taus': taus,  # ndarray 转 list
    'traj_nn_in_1': traj_nn_in_1,  # 若为 ndarray
    'traj_nn_in_2': traj_nn_in_2,  # 若为 ndarray
    # 'traj_nn_out': [float(v) if isinstance(v, np.ndarray) else v for v in traj_nn_out],  # 单层转 float
    'traj_nn_out': [float(np.asarray(v).squeeze()) for v in traj_nn_out],
    'traj_pos': traj_pos.tolist(),
    'traj_vel': traj_vel.tolist()
}

# Output JSON to stdout (can be captured by MATLAB)
print(json.dumps(diagInfo))