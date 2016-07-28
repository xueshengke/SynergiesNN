function mnistNN
clear all; clc;
addpath NN;
addpath util;
load mnist_uint8;

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%% normalize and mean
train_x = train_x - repmat(mean(train_x, 1), [size(train_x, 1) 1]);
test_x = test_x - repmat(mean(test_x, 1), [size(test_x, 1) 1]);
train_x = train_x - repmat(mean(train_x, 2), [1 size(train_x, 2)]);
test_x = test_x - repmat(mean(test_x, 2), [1 size(test_x, 2)]);

% [train_x, mu, sigma] = zscore(train_x);
% test_x = normalize(test_x, mu, sigma);

%% display train and test data structure
imagesize = sqrt(size(train_x, 2));
trainNumber = size(train_x, 1);
testNumber = size(test_x, 1);
fprintf('prepare train data, %d * %d, %d\n', imagesize, imagesize, trainNumber);
fprintf('prepare test data, %d * %d, %d\n', imagesize, imagesize, testNumber);

%% initiate a neural net
%rand('state',0);
nn = nnsetup([784 200 10]);
nn.weightPenaltyL2         = 1e-4;         % L2 weight decay
nn.momentum                = 0.9 ;           % momentum
nn.dropoutFraction         = 0 ;           % dropout fraction 
% nn.scaling_learningRate  = 1;            % scaling factor for the learning rate (each epoch)
% nn.nonSparsityPenalty    = 0;            % non sparsity penalty
nn.activation_function     = 'adapt_sigm'; % sigmoid activation function : tanh_opt or sigm
nn.output                  = 'adapt_sigm';  % use logistic or softmax or linear output
nn.sigm_learningRate       = 0.02 ;         % adjustable sigm parameters' learning rate
nn.learningRate            = 1 ;          % sigm require a lower learning rate
nn.scaling                 = 0.991;      % scaling factor for the learning rate (each epoch)

opts.plot      = 1;      % enable plotting
opts.numepochs = 400;    % number of full sweeps through data
opts.batchsize = 600;   % take a mean gradient step over this many samples

% nn.plotAlphaHidden = zeros(opts.numepochs, 1);
% nn.plotAlphaOutput = zeros(opts.numepochs, 1);
% nn.plotBetaHidden  = zeros(opts.numepochs, 1);
% nn.plotBetaOutput  = zeros(opts.numepochs, 1);
% nn.plotEntropyHidden = zeros(opts.numepochs, 1);
% nn.plotEntropyOutput = zeros(opts.numepochs, 1);

%% train the nerual net
[nn, L] = nntrain(nn, train_x, train_y, opts);

%% test the neural net
[er, bad] = nntest(nn, test_x, test_y);
fprintf('final test accuracy: %.3f %%\n', 100 * (1 - er));

%assert(er < 0.08, 'Too big error');
end
