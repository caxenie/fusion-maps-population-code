% sigmoid function computation for intermediate layer neurons activation
% computation 
% v0 = maximum limit for the function
% u0 = switch point (changes polarity)
function psi = sigmoid(v0, u0, u)
     psi = v0*(1/(1 + exp(u0 - u)));
end