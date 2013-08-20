% sigmoid function computation for intermediate layer neurons activation
% computation 
function psi = sigmoid(u)
    v0 = 10;
    u0 = 20;
    psi = v0*(1+1/exp(u0-u));
end;