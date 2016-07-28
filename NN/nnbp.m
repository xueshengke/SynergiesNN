function nn = nnbp(nn)
% NNBP performs backpropagation
% nn = nnbp(nn) returns a neural network structure with updated weights 
    
    n = nn.layer;
    sparsityError = 0;
    switch nn.output
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n}));
        case 'tanh_opt'
            d{n} = - nn.e .* (1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{n} .^ 2));
        case {'softmax','linear'}
            d{n} = - nn.e;
        case 'adapt_sigm'
            d{n} = - nn.e .* dAdaptSigm(nn, n, nn.a{n});  
        case 'adapt_tanh'
            d{n} = - nn.e .* dAdaptTanh(nn, n, nn.a{n});            
    end
    
    for i = (n - 1) : -1 : 2
        % derivative of the activation function
        switch nn.activation_function 
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i});
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i} .^ 2);
            case  'adapt_sigm'
            	d_act = dAdaptSigm(nn, i, nn.a{i}(:, 2 : end));   
                d_act = [zeros(size(nn.a{i}, 1), 1) d_act];
            case  'adapt_tanh'
            	d_act = dAdaptTanh(nn, i, nn.a{i}(:, 2 : end));   
                d_act = [zeros(size(nn.a{i}, 1), 1) d_act];             
        end
        
        if(nn.nonSparsityPenalty > 0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty ...
                * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % backpropagate first derivatives
        if i + 1 == n % in this case, d{n} has no bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:, 2 : end) * nn.W{i} + sparsityError) .* d_act;
        end
        
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end

    end

    for i = (n - 1) : -1 : 1
        if i + 1 == n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);
        else
            nn.dW{i} = (d{i + 1}(:, 2 : end)' * nn.a{i}) / size(d{i + 1}, 1);             
        end
        
        switch nn.activation_function 
            case 'adapt_sigm'
                if i + 1 == n
                    x = nn.z{i+1};  y = nn.a{i+1};
                else
                    x = nn.z{i+1};  y = nn.a{i+1}(:, 2 : end);
                end
                nn.sigmPara{i+1}.dAlpha = 1 ./ nn.sigmPara{i+1}.alpha ...
                    + mean(x, 1) - 2 * mean(x .* y, 1);
                nn.sigmPara{i+1}.dBeta  = 1 - 2 * mean(y, 1);
%                 for j = 1 : nn.size(i + 1)
%                     nn.sigmPara{i}{j}.d_alpha = 1 / nn.sigmPara{i}{j}.alpha + mean(nn.z{i+1}(:, j)) - 2 * mean(nn.z{i+1}(:, j) .* nn.a{i+1}(:, j));
%                     nn.sigmPara{i}{j}.d_beta = 1 - 2 * mean(nn.a{i+1}(:, j));
%                 end
            case 'adapt_tanh'
                if i + 1 == n
                    x = nn.z{i+1};  y = nn.a{i+1};
                else
                    x = nn.z{i+1};  y = nn.a{i+1}(:, 2 : end);
                end
                nn.sigmPara{i+1}.dAlpha = 1 ./ nn.sigmPara{i+1}.alpha ...
                    - 2 * mean(x .* y, 1);
                nn.sigmPara{i+1}.dBeta  = - 2 * mean(y, 1);                
%                 for j = 1 : nn.size(i + 1)
%                     nn.sigmPara{i}{j}.d_alpha = 1 / nn.sigmPara{i}{j}.alpha - 2 * mean(nn.z{i+1}(:, j) .* nn.a{i+1}(:, j));
%                     nn.sigmPara{i}{j}.d_beta = - 2 * mean(nn.a{i+1}(:, j));    
%                 end
        end        
    end
end
