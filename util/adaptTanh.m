function [ weightedSum, activation ] = adaptTanh( nn, layer, input, W )
% nn       neural network
% input    activation of previous layer
% W        weight connection

% weightedSum = zeros(size(input, 1), size(W, 2));
% activation = zeros(size(input, 1), size(W, 2));

weightedSum = input * W ;
u = weightedSum .* repmat(nn.sigmPara{layer}.alpha, [size(input, 1) 1]) ...
    + repmat(nn.sigmPara{layer}.beta, [size(input, 1) 1]) ;
activation  = ( exp(u) - exp(-u) ) ./ ( exp(u) + exp(-u) );

% for i = 1 : size(W, 2)
%     column = input * W(:, i);
%     weightedSum(:, i) = column;
%     alpha = nn.sigmPara{layer - 1}{i}.alpha;
%     beta  = nn.sigmPara{layer - 1}{i}.beta;
%     u = alpha * column + beta;
%     column = ( exp(u) - exp(-u) ) ./ ( exp(u) + exp(-u) );
%     activation(:, i) = column;
% end
    
end

