# OPERA

This repository contains the artifact for the paper *"Repair of DNN Control Policies via STL-Guided Patch Synthesis"*.

## Abstract

Recently, there comes an increasing demand in employing deep neural networks (DNNs) for control of cyber-physical systems (CPS), giving birth to  emerging applications of AI-enabled CPS, including autonomous driving, medical devices, 
robotics and etc. Despite such prevalence, there arises a surge of concerns about the safety of those systems, due to the opaque decision logic of DNNs, and therefore, effective safety assurance techniques are needed to ensure their 
safe functioning in their deployment environments. In the loop of safety assurance, repair is a necessary step, which aims to automatically fix the unsafe behaviors of the system after finding them out. While there have been extensive research
on repair of DNN models, these works either target standalone DNN models (e.g., image classifiers), or require white-box model information; as a consequence, they may not be well-suited for repair of DNN control policies.

In this paper, we tackle the problem of repairing DNN control policies, in order to enhance the safety of the whole system. To this end, we propose a framework called Opera that  can automatically synthesize control patches used 
to correct the unsafe control decisions, guided by system-level specifications defined in signal temporal logic (STL). Given an unsafe system execution, Opera first localizes the causal segments needed to be repaired; 
then, for each unsafe execution, Opera searches for a patch of the control policy, that can ensure the safe behavior of the system. To repair the system, Opera adds these safe executions to the training set and retrains the DNN controller. 
We conduct a comprehensive evaluation for Opera on six systems spanning over different safety-critical domains. Evaluation results show that Opera can effectively repair the critical safety issues in DNN controllers, 
thereby improving system reliability. 

## System requirement

- Operating system: Linux or MacOS;

- Matlab (Simulink/Stateflow) version: >= 2020a. (Matlab license needed)

- Python version: >= 3.3

- MATLAB toolboxes dependency: 
  1. [Simulink](https://www.mathworks.com/products/simulink.html) for ACC, AFC, and WT.
  2. [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html) for all benchmarks.
  3. [Model Predictive Control Toolbox](https://www.mathworks.com/products/model-predictive-control.html) for ACC.
  4. [Signal Processing Toolbox](https://www.mathworks.com/products/signal.html) for WT.
  5. [DSP System Toolbox](https://www.mathworks.com/products/dsp-system.html) for WT.

## Installation

- Install [Breach](https://github.com/decyphir/breach)
  1. start Matlab, set up a C/C++ compiler using the command `mex -setup`. (Refer to [here](https://www.mathworks.com/help/matlab/matlabexternal/changing-default-compiler.html) for more details.)
  2. navigate to `breach/` in Matlab commandline, and run the command `InstallBreach`.
- Install [Git Large File Storage](https://git-lfs.com)
  1. for Linux:
     ```
     curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
     sudo apt-get install git-lfs
     git lfs install
     ```
  2. for MacOS:
     ```
     brew install git-lfs
     git lfs install
     ```
  3. Retrieve the execution traces for repair and retraining stored in the form of large files by the following command `git lfs pull`.
  
 ## Usage

 To reproduce the experimental results, users should follow the steps below:
 
  - The user-specified configuration files are stored under the directory `test/config/`. Replace the paths of `OPERA` in user-specified file under the line `addpath 1` with their own path. Users can also specify other configurations, such as max_gen and pop_size.
  - Users need to edit the executable scripts permission using the command `chmod -R 777 *`.
  - The corresponding log will be stored under directory `output/`.
 
 ### Signal Diagnosis and Repair

 - Navigate to the directory `test/`. Run the command `python autorepair.py config/[benchmark]/repair`.
 - Now the executable scripts have been generated under the directory `test/scripts/`. 
 - Navigate to the root directory `OPERA/` and run the command `make`. The automatically generated datasets file for NN controller retraining and chromosome files will be stored under the root directory.
 
 ### NN Controller Retraining
 
  - Navigate to the directory `test/`. Run the command `python train.py config/[benchmark]/train`.
  - Now the executable scripts have been generated under the directory `test/scripts/`. 
  - Navigate to the root directory `OPERA/` and run the command `make`. The automatically generated NN controller files will be stored under the root directory.
 
 ### Evaluation
 
  - Navigate to the directory `test/`. Run the command `python bef_eval_[benchmark].py config/[benchmark]/bef_eval` and `python rep_eval_[benchmark].py config/[benchmark]/rep_eval`.
  - Now the executable scripts have been generated under the directory `test/scripts/`.
  - Navigate to the root directory `OPERA/` and run the command `make`. The automatically generated evaluation results will be stored under the root directory.


 
