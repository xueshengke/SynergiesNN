function nn = nnff(nn, x, y)
% NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns a neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.layer;
    m = size(x, 1);
    
    % add the bias term
    x = [zeros(m,1) x];
    nn.a{1} = x;

    % feedforward pass
    for i = 2 : n - 1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.z{i} = nn.a{i - 1} * nn.W{i - 1}' ;
                nn.a{i} = sigm(nn.z{i});
            case 'tanh_opt'
                nn.z{i} = nn.a{i - 1} * nn.W{i - 1}' ;                
                nn.a{i} = tanh_opt(nn.z{i});
            case 'adapt_sigm'
                [ nn.z{i}, nn.a{i} ] = adaptSigm(nn, i, nn.a{i - 1}, nn.W{i - 1}');
            case 'adapt_tanh'
                [ nn.z{i}, nn.a{i} ] = adaptTanh(nn, i, nn.a{i - 1}, nn.W{i - 1}');
        end
        
        % dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        % calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        % add the bias term
        nn.a{i} = [zeros(m,1) nn.a{i}];
    end
    switch nn.output 
        case 'sigm'
            nn.z{n} = nn.a{n - 1} * nn.W{n - 1}' ;
            nn.a{n} = sigm(nn.z{n});
        case 'tanh_opt'
            nn.z{n} = nn.a{n - 1} * nn.W{n - 1}' ; 
            nn.a{n} = tanh_opt(nn.z{n});
        case 'linear'
            nn.z{n} = nn.a{n - 1} * nn.W{n - 1}' ; 
            nn.a{n} = nn.z{n};
        case 'softmax'
            nn.z{n} = nn.a{n - 1} * nn.W{n - 1}' ;
            nn.a{n} = exp(bsxfun(@minus, nn.z{n}, max(nn.z{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
        case 'adapt_sigm'
            [ nn.z{n}, nn.a{n} ] = adaptSigm(nn, n, nn.a{n - 1}, nn.W{n - 1}');
        case 'adapt_tanh'
            [ nn.z{n}, nn.a{n} ] = adaptTanh(nn, n, nn.a{n - 1}, nn.W{n - 1}');
    end

    % error and loss
    nn.e = y - nn.a{n};
    
    switch nn.output
        case {'sigm', 'tanh_opt', 'linear', 'adapt_sigm', 'adapt_tanh'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
