% compute the Gaussian tunning curve intersection with a given value
function fval = gauss_val(val, pref, sigma, limit)
    % check if value in the limits of the tuning curve
   if ((val<limit) && (val>-limit))
    fval = 1/(sqrt(2*pi)*sigma)*exp(-(val - pref)^2/(2*sigma*sigma));
   else
    fval = limit*2; % something out-of-range
   end
end