% unimodal function for the mapping from the intermediate layer of the net
% to the output population 
function out = G_map(x)
    scale = 100;
    sigma = 20;
    x_peak = 0;
    out = scale*exp(-(x - x_peak)^2/(2*sigma*sigma));
end