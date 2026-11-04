function trainNNCtrl(bench, XTrain, YTrain, nn_type, nn_stru, algo, is_nor, varargin)
% trainNNCtrl function is the main function to train an NN controller based on supervised learning.
%
% Initial train (9 inputs):
%   trainNNCtrl(bench, XTrain, YTrain, nn_type, nn_stru, algo, is_nor, sti, date)
%
% Retrain (12 inputs):
%   trainNNCtrl(bench, XTrain, YTrain, nn_type, nn_stru, algo, is_nor, epoch, ratio, sti, date, ni)
%
% Inputs:
%   bench: benchmark name used for naming the NN controller and its derived net file, for example, 'ACC'
%   XTrain: input data X, collected from the input of the original controller during system execution
%   YTrain: target output data Y, collected from the output of the original controller during system execution
%   nn_type: the type of the NN controller, FFNN or LRNN
%   nn_stru: NN controller structure
%   algo: training algorithm, for example, 'trainscg'
%   is_nor: whether XTrain and YTrain has undergone normalization or not
%   sti: strategy index
%   date: date of training the neural network
%   epoch: (retrain only) maximum training epochs
%   ratio: (retrain only) training data ratio, used in naming when not -1
%   ni: (retrain only) nn controller index
% Outputs:
%   NN controller and its derived net file

if nargin == 9
    sti = varargin{1};
    date = varargin{2};
    is_initial_train = true;
elseif nargin == 12
    epoch = varargin{1};
    ratio = varargin{2};
    sti = varargin{3};
    date = varargin{4};
    ni = varargin{5};
    is_initial_train = false;
else
    error('trainNNCtrl: expected 9 args (initial train) or 12 args (retrain), got %d.', nargin);
end

% if you want to maintain the reproducibility of the NN controller, you can fix the random seed using the command: rng("default");
% rng("default")

% initialize the generator seed based on the current time
rng("shuffle");

% for naming the NN controller and its derived net file
if is_nor == 1
    is_nor_str = '_Nor_';
elseif is_nor == 0
    is_nor_str = '_UnNor_';
else
    error('Check the value of is_nor!');
end

if strcmpi(nn_type, "FFNN")
    % set the hidden layers based on nn_stru
    hiddenLayers = nn_stru;
    % creat a feedforward neural network, the default algo is 'trainlm' (I personally also recommend using this), 
    % but the official version of 'trainlm' does not support GPU acceleration
    ffnn_net = feedforwardnet(hiddenLayers, algo);
    % you can also specify the algo by the command: ffnn_net.trainFcn = 'trainscg';
    for i = 1:length(hiddenLayers)
        % type 'help nntransfer' to select a activation function. Here, we use 'ReLU'
        ffnn_net.layers{i}.transferFcn = 'poslin';
    end

    % set the activation function of last layer as 'purelin', a linear transfer function.
    ffnn_net.layers{length(hiddenLayers)+1}.transferFcn = 'purelin';

    % for input and output, mapminmax min-max normalization
    ffnn_net.inputs{1}.processFcns = {};
    ffnn_net.outputs{length(hiddenLayers)+1}.processFcns = {};
    
    ffnn_net = configure(ffnn_net, XTrain, YTrain);
    ffnn_net.plotFcns = {'plotperform', 'plottrainstate',...
        'ploterrhist', 'plotregression', 'plotresponse'};

    [xs,xi,ai,ts] = preparets(ffnn_net, XTrain, YTrain);
    
    % The commented-out lines below are the default hyperparameters used
    % for 'trainlm', with some being specific to 'trainlm' (e.g, mu, mu_dec, mu_inc, mu_max), 
    % with some being common to all algos (showWindow, showCommandLine, show, epochs, time, goal, min_grad, max_fail)
    % You can specify algorithm-specific parameters by uncommenting the following lines, 
    % please refer to https://www.mathworks.com/help/deeplearning/ug/choose-a-multilayer-neural-network-training-function.html
    % ffnn_net.trainParam.showWindow = 'true';          % Show Training Window Feedback 
    % ffnn_net.trainParam.showCommandLine = 'false';    % Show Command Line Feedback
    % ffnn_net.trainParam.show = 25;                    % Command Line Frequency
    if ~is_initial_train
        ffnn_net.trainParam.epochs = epoch;             % Maximum Epochs
    end
    % ffnn_net.trainParam.time = Inf;                   % Maximum Training Time 
    % ffnn_net.trainParam.goal = 0;                     % Performance Goal
    ffnn_net.trainParam.min_grad = 1e-10;               % Minimum Gradient. The default value is 1e-07. Here, I set the min_grad as 1e-10.
    % ffnn_net.trainParam.max_fail = 6;                 % Maximum Validation Checks 
    % ffnn_net.trainParam.mu = 0.001;                   % Initial Mu
    % ffnn_net.trainParam.mu_dec = 0.1;                 % Mu Decrease Ratio
    % ffnn_net.trainParam.mu_inc = 10;                  % Mu Increase Ratio 
    % ffnn_net.trainParam.mu_max = 10000000000;         % Maximum Mu

    % start training an FFNN controller
    [ffnn_net, tr] = train(ffnn_net, xs, ts, xi, ai, 'showResources', 'yes', 'useParallel', 'yes', 'useGPU', 'no');

    % obtain the predicted value Y
    Y = ffnn_net(xs, xi, ai);
    % assess the performance of the trained NN controller. The default performance function is mean squared error.
    ffnn_perf = perform(ffnn_net, Y, ts);
    ffnn_error = cell2mat(Y) - cell2mat(ts);
    % plot(ffnn_perf);
    % plot(ffnn_error);

    % generate simulink block for ffnn simulation
    if sti == 0
        if is_initial_train
            nn_ctr_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_")  + "_ctr" + is_nor_str + date;
            nn_conf_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_config" + is_nor_str + date;
        else
            nn_ctr_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_")  + "_ctr" + is_nor_str + date + '_' + num2str(ni);
            nn_conf_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_config" + is_nor_str + date + '_' + num2str(ni);
        end
    elseif ratio == -1
        % for epoch
        nn_ctr_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_")  + "_st_" + sti + "_ctr" + is_nor_str + "Ep_" + num2str(epoch) + '_' + date + '_' + num2str(ni);
        nn_conf_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_st_" + sti + "_config" + is_nor_str + "Ep_" + num2str(epoch) + '_' + date + '_' + num2str(ni);
    else
        % for ratio
        nn_ctr_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_")  + "_st_" + sti + "_ctr" + is_nor_str + "R_" + num2str(ratio) + '_' + date + '_' + num2str(ni);
        nn_conf_name = bench + "_FFNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_st_" + sti + "_config" + is_nor_str + "R_" + num2str(ratio) + '_' + date + '_' + num2str(ni);
    end
    
    nn_ctr_block = gensim(ffnn_net, 'Name', nn_ctr_name);
    save_system(nn_ctr_block);
    close_system(nn_ctr_block);
    % save the ffnn net file
    net = ffnn_net;
    save(nn_conf_name + ".mat", 'net');
elseif strcmpi(nn_type, "LRNN")
    % the default input delays is 1:2. In RNN, there is an additional input
    % of the tapped delay line vector. Here, we use a tapped delay line with delays from 1 to 2.
    % Refer to https://www.mathworks.com/help/deeplearning/ug/design-time-series-time-delay-neural-networks.html for more details.
    layerDelays = 1:2;
    % set hidden layers based on nn_stru, the default hiddenLayers is [10] 
    hiddenLayers = nn_stru;
    % create a layer recurrent neural network, the default algo is 'trainlm'
    lrnn_net = layrecnet(layerDelays, hiddenLayers, algo);
    lrnn_net.inputs{1}.processFcns = {};
    lrnn_net.outputs{length(hiddenLayers)+1}.processFcns = {};
    [xs, xi, ai, ts] = preparets(lrnn_net, XTrain, YTrain);

    % The commented-out lines below are the default hyperparameters used
    % for 'trainlm', with some being specific to 'trainlm' (e.g, mu, mu_dec, mu_inc, mu_max), 
    % with some being common to all algos (showWindow, showCommandLine, show, epochs, time, goal, min_grad, max_fail)
    % You can specify algorithm-specific parameters, 
    % please refer to https://www.mathworks.com/help/deeplearning/ug/choose-a-multilayer-neural-network-training-function.html
    % lrnn_net.trainParam.showWindow = 'true';          % Show Training Window Feedback 
    % lrnn_net.trainParam.showCommandLine = 'false';    % Show Command Line Feedback
    % lrnn_net.trainParam.show = 25;                    % Command Line Frequency
    % lrnn_net.trainParam.epochs = 1000;                % Maximum Epochs
    % lrnn_net.trainParam.time = Inf;                   % Maximum Training Time 
    % lrnn_net.trainParam.goal = 0;                     % Performance Goal
    % lrnn_net.trainParam.min_grad = 1e-07;             % Minimum Gradient 
    % lrnn_net.trainParam.max_fail = 6;                 % Maximum Validation Checks 
    % lrnn_net.trainParam.mu = 0.001;                   % Initial Mu
    % lrnn_net.trainParam.mu_dec = 0.1;                 % Mu Decrease Ratio
    % lrnn_net.trainParam.mu_inc = 10;                  % Mu Increase Ratio 
    % lrnn_net.trainParam.mu_max = 10000000000;         % Maximum Mu

    % start training an LRNN controller, tr is the training record
    [lrnn_net, tr] = train(lrnn_net, xs, ts, xi, ai, 'showResources', 'yes', 'UseParallel', 'yes', 'UseGPU', 'no');

    % obtain the predicted value Y
    Y = lrnn_net(xs, xi, ai);
    % assess the performance of the trained NN controller. The default performance function is mean squared error.
    lrnn_perf = perform(lrnn_net,Y,ts);
    lrnn_error = cell2mat(Y) - cell2mat(ts);
    % plot(lrnn_perf);
    % plot(lrnn_error);

    % generate simulink block for lrnn simulation
    if is_initial_train
        nn_ctr_name = bench + "_LRNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_ctr" + is_nor_str + date;
        nn_conf_name = bench + "_LRNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_config" + is_nor_str + date;
    else
        nn_ctr_name = bench + "_LRNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_ctr" + is_nor_str + date + '_' + num2str(ni);
        nn_conf_name = bench + "_LRNN_" + algo + "_" + strjoin(string(nn_stru), "_") + "_config" + is_nor_str + date + '_' + num2str(ni);
    end
    nn_ctr_block = gensim(lrnn_net, 'Name', nn_ctr_name);
    save_system(nn_ctr_block);
    close_system(nn_ctr_block);
    % save the lrnn net file
    net = lrnn_net;
    save(nn_conf_name + ".mat", 'net');
else
    error('Check your nn_type!');
end
end