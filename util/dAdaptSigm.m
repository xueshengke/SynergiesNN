function [ derivative ]  = dAdaptSigm(nn, layer, activation)
    derivative = activation .* (1 - activation);
    derivative = derivative .* repmat(nn.sigmPara{layer}.alpha, [size(activation, 1) 1]);
    
%     for i = 1 : size(activation, 2)
%         alpha = nn.sigmPara{layer - 1}{i}.alpha;
%         derivative(:, i) = alpha * derivative(:, i);
%     end
end