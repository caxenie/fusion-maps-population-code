% Gaussian tuning curve parametrizable for the neurons in the population
function [x, fx]= gauss_tuning(pref, sigma, limit)
    % limit is the range (+/-)
    x = -limit:limit;
    % compute the gaussian
    fx = 1/(sqrt(2*pi)*sigma)*exp(-(x-pref).^2/(2*sigma*sigma));
end