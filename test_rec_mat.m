% recurrent conection test
We      = 1.6; % short range excitation strength We > Wi
Wi      = 0.6; % long range inhibition strength 
sigma_e = 10;  % excitation Gaussian profile sigma_e < sigma_i
sigma_i = 30;  % inhibiton Gaussian profile
W = zeros(neurons_complete_x, neurons_complete_y);

% build the recurrent connectivity matrix
for i=1:neurons_complete_x
    for j=1:neurons_complete_y
        for k = 1:neurons_complete_x
            for l = 1:neurons_complete_y
                W(i,j) = We*(exp(-((i-k)^2+(j-l)^2)/(2*sigma_e^2))) - ...
                         Wi*(exp(-((i-k)^2+(j-l)^2)/(2*sigma_i^2)));
            end
        end
    end
end
mesh(1:neurons_complete_y, 1:neurons_complete_x, W(1:neurons_complete_y, 1:neurons_complete_x))

    