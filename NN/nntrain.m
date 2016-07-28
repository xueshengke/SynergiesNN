function [nn, L]  = nntrain(nn, train_x, train_y, opts, val_x, val_y)
% NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);
layer = nn.layer;

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs * numbatches, 1);
nn.entropy = zeros(numepochs * numbatches, layer);

n = 1;
count = 1;
lastTrainErr = 0;
thisTrainErr = 0;

for i = 1 : numepochs
    tic;
    
    randNum = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(randNum((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(randNum((l - 1) * batchsize + 1 : l * batchsize), :);
        
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);
            
%         nn.plotAlphaHidden(n) = nn.sigmPara{1}{1}.alpha;
%         nn.plotBetaHidden(n)  = nn.sigmPara{1}{1}.beta;        
%         nn.plotAlphaOutput(n) = nn.sigmPara{2}{1}.alpha;
%         nn.plotBetaOutput(n)  = nn.sigmPara{2}{1}.beta;
%         nn.plotEntropyHidden(n) = entropy(nn.a{2}(1, 2 : end));
%         nn.plotEntropyOutput(n) = entropy(nn.a{3}(1, 2 : end));
        
        L(n) = nn.L;
        
        for k = 2 : layer
           nn.entropy(n, k) = entropy(nn.a{k});
        end
        
        n = n + 1;
    end
    
    if i > 2
       thisTrainErr =  mean(L((n-numbatches):(n-1)));
    end
%     if i > 10 && abs(L(n-1) - L(n-numbatches)) < 2e-4
    if i > 10 && abs(lastTrainErr - thisTrainErr) < 1e-4
       nn = nnapplysigms(nn);
       fprintf('sigm parameters adjusted %d times.\n', count);
       count = count + 1;
    end  
    
    lastTrainErr = thisTrainErr;
    
    t = toc;

    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train MSE = %f, val mse = %f', ...
            loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
        
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ...
        '. Took ' num2str(t) ' seconds' '. Mini-batch MSE on training set is ' ...
        num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
    
    % annealing learning rate, update here
    nn.learningRate = nn.learningRate * nn.scaling;
    nn.sigm_learningRate = nn.sigm_learningRate * nn.scaling;
end
end

