% sigmoid function computation for intermediate layer neurons activation
% computation 
function psi = sigmoid(u)
    v0 = 0.1;
    u0 = 0.2;
    psi = v0*(1+1/exp(u0-u));
end