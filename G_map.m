% unimodal function for the mapping from the intermediate layer of the net
% to the output population 
function out = G_map(x)
    scale = 80;
    sigma = 10;
    out = scale*exp(-(x)^2/(2*sigma*sigma));
end