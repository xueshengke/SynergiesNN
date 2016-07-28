function nn = nnapplysigms(nn)
% NNAPPLYSIGMS updates sigmoid function parameters 
% nn = nnapplysigms(nn) returns a neural network structure with updated
% sigm parameters
func = nn.activation_function;
if ~(strcmp(func, 'adapt_sigm') || strcmp(func, 'adapt_tanh'))
    return ;
end

for i = 1 : (nn.layer - 1)
%     if nn.weightPenaltyL2 > 0
%         dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i}, 1), 1) nn.W{i}(:, 2 : end)];
%     else
%         dW = nn.dW{i};
%     end
% 
%     dW = nn.learningRate * dW;
% 
%     if(nn.momentum>0)
%         nn.vW{i} = nn.momentum * nn.vW{i} + dW;
%         dW = nn.vW{i};
%     end
% 
%     nn.W{i} = nn.W{i} - dW;

%     switch nn.activation_function
%         case {'adapt_sigm', 'adapt_tanh'}
    nn.sigmPara{i+1}.alpha = nn.sigmPara{i+1}.alpha - nn.sigm_learningRate * nn.sigmPara{i+1}.dAlpha;
    nn.sigmPara{i+1}.beta  = nn.sigmPara{i+1}.beta  - nn.sigm_learningRate * nn.sigmPara{i+1}.dBeta;
%             for j = 1 : nn.size(i + 1)
%                 nn.sigmPara{i}{j}.alpha = nn.sigmPara{i}{j}.alpha + nn.sigm_learningRate * nn.sigmPara{i}{j}.d_alpha;
%                 nn.sigmPara{i}{j}.beta = nn.sigmPara{i}{j}.beta + nn.sigm_learningRate * nn.sigmPara{i}{j}.d_beta;
%             end
%     end
%     fprintf('layer : %d, alpha : %.2f , beta : %.2f \n', ...
%         i + 1, nn.sigmPara{i}{1}.alpha, nn.sigmPara{i}{1}.beta);

end

end
