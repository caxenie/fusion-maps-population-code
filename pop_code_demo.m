%% Demo software usign population code for estimating arbitrary functions
% 
% the setup contains 2 input populations each coding some scalar
% (unimodal) variable which are projected onto a 2D network of units with
% neurons exhibiting short range excitation and long range inhibition
% dynamics
% the ouput from the intermediate layer is projected to an output
% population 
% the network has no explicit input and output as each of the populations
% may be considered inputs / outputs and the processing happens in the
% intermediate layer

%% INITIALIZATION
clear all;
clc; close all;

% define the 1D populations (symmetric (size wise) populations)
neurons_pop_x = 50;
neurons_pop_y = 50;
neurons_pop_z = 50;
neurons_complete_x = neurons_pop_x + 1;  % number of neurons in the input populations
neurons_complete_y = neurons_pop_y + 1;
neurons_complete_z = neurons_pop_z + 1;
noise_scale = 7;

%% HACK polarity change in sigmoid activation function
sigmoid_polarity_switch = 1.1765;

% neuron information (neuron index i)
%   i       - index in the population 
%   ri      - activity of the neuron (firing rate)
%   fi      - tuning curve (e.g. Gaussian)
%   vi      - preferred value - even distribution in the range
%   etai    - neuronal noise value - zero mean and tipically correlated

% scaling factor for tuning curves
bkg_firing = 10; % spk/s - background firing rate
scaling_factor = 80; % motivated by the typical background and upper spiking rates
max_firing = 100;

% demo params 
encoded_val_x = -45;
encoded_val_y = 20;
encoded_val_z = 54;

%% generate first population and initialize
% preallocate
vix=zeros(1, neurons_complete_x);

% zero mean neuronal noise 
etax = randn(neurons_complete_x, 1)*noise_scale;
% population standard deviation -> coarse (big val) / sharp (small val) receptive field
sigma_x = 10;
% population range of values (+/-)
x_pop_range = 100;
% peak to peak spacing in tuning curves
x_spacing = x_pop_range/((neurons_complete_x-1)/2);

% init population
for i=1:neurons_complete_x
    % evenly distributed preferred values in the interval
    vix(i) = -x_pop_range+(i-1)*(x_pop_range/((neurons_complete_x-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(vix(i), ...
                               sigma_x, ...
                               x_pop_range, ...
                               scaling_factor);
    fix(i).p = pts;
    fix(i).v = vals;
    x_population(i) = struct('i',   i, ...
                            'vi',   vix(i),...
                            'fi',   fix(i), ...
                            'etai', etax(i),...
                            'ri',   abs(randi([bkg_firing , max_firing])));
end;

%% generate second population and initialize
% preallocate
viy=zeros(1, neurons_complete_y);
 
% zero mean neuronal noise 
etay = randn(neurons_complete_y, 1)*noise_scale;
% population standard deviation - coarse (big val) / sharp receptive field
sigma_y = 10;
% population range of values (+/-)
y_pop_range = 100;
% peak to peak spacing in tuning curves
y_spacing = y_pop_range/((neurons_complete_y-1)/2);
% init population
for i=1:neurons_complete_y
    % evenly distributed preferred values in the interval
    viy(i) = -y_pop_range+(i-1)*(y_pop_range/((neurons_complete_y-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(viy(i), ...
                               sigma_y, ...
                               y_pop_range, ...
                               scaling_factor);
    fiy(i).p = pts;
    fiy(i).v = vals;
    y_population(i) = struct('i',   i, ...
                            'vi',   viy(i),...
                            'fi',   fiy(i), ...
                            'etai', etay(i),...
                            'ri',   abs(randi([bkg_firing , max_firing])));
end;

%% generate third population and initialize
% preallocate
viz=zeros(1, neurons_complete_z);
 
% zero mean neuronal noise 
etaz = randn(neurons_complete_z, 1)*noise_scale;
% population standard deviation - coarse (big val) / sharp receptive field
sigma_z = 10;
% population range of values (+/-) 
z_pop_range = 200;
% peak to peak spacing in tuning curves
z_spacing = z_pop_range/((neurons_complete_z-1)/2);
% init population
for i=1:neurons_complete_z
    % evenly distributed preferred values in the interval
    viz(i) = -z_pop_range+(i-1)*(z_pop_range/((neurons_complete_z-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(viz(i), ...
                               sigma_z, ...
                               z_pop_range, ...
                               scaling_factor);
    fiz(i).p = pts;
    fiz(i).v = vals;
    z_population(i) = struct('i',   i, ...
                            'vi',   viz(i),...
                            'fi',   fiz(i), ...
                            'etai', etaz(i),...
                            'ri',   randi([bkg_firing , max_firing]));
end;
%% VISUALIZATION
figure;
set(gcf,'color','w');

% first population 
% plot the tunning curves of all neurons for the first population
subplot(6, 4, [1 2]);
for i=1:neurons_complete_x
    plot(x_population(i).fi.p, x_population(i).fi.v);
    hold all;
end;
grid off;
set(gca, 'Box', 'off');
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');

% plot the encoded value in the population
subplot(6, 4, [5 6]);
% make the encoding of the value in the population and add noise
% noise on every trial to get trial-to-trial variability
% zero mean noise
etax = randn(neurons_complete_x, 1)*noise_scale;
etay = randn(neurons_complete_y, 1)*noise_scale;
etaz = randn(neurons_complete_z, 1)*noise_scale;

for i=1:neurons_complete_x
    % scale the firing rate to proper values and compute fi
    x_population(i).ri = gauss_val(encoded_val_x, ...
                                   x_population(i).vi, ...
                                   sigma_x, ...
                                   scaling_factor) + ...
                                   etax(i);
    % rate should be positive althought noise can make small vallues
    % negative
    x_population(i).ri = abs(x_population(i).ri);
end;
% plot the noisy hill of population activity encoding the given value
% index for neurons
j = 1;
for i=-x_pop_range:x_pop_range
    % display on even spacing of the entire input domain
    if(rem(i, x_spacing)==0)
        plot(i, x_population(j).ri, 'o');
        hold all;
        j = j+1;
    end;
end; 
grid off;
set(gca, 'Box', 'off');
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_x));
ylabel('Activity (spk/s)');
xlabel('Preferred value');

% second population 
subplot(6, 4, [3 4]);
for i=1:neurons_complete_y
    plot(y_population(i).fi.p, y_population(i).fi.v);
    hold all;
end;
grid off;
set(gca, 'Box', 'off');
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');

% plot the encoded value in the population
subplot(6, 4, [7 8]);
% make the encoding of the value in the population and add noise
for i=1:neurons_complete_y
    % scale the firing rate to proper values and compute fi
    y_population(i).ri = gauss_val(encoded_val_y, ...
                                   y_population(i).vi, ...
                                   sigma_y, ...
                                   scaling_factor) + ...
                                   etay(i);
    % rate should be positive althought noise can make small vallues
    % negative
    y_population(i).ri = abs(y_population(i).ri);                               
end;
% plot the noisy hill of population activity encoding the given value
% index for neurons
j = 1;
for i=-y_pop_range:y_pop_range
    % display on even spacing of the entire input domain
    if(rem(i, y_spacing)==0)
        plot(i, y_population(j).ri, 'o');
        hold all;
        j = j+1;
    end;
end;
grid off;
set(gca, 'Box', 'off');      
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_y));
ylabel('Activity (spk/s)');
xlabel('Preferred value');

% third population 
subplot(6, 4, [18 19]);
for i=1:neurons_complete_z
    plot(z_population(i).fi.p, z_population(i).fi.v);
    hold all;
end;
grid off;
set(gca, 'Box', 'off');
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');

% plot the encoded value in the population
subplot(6, 4, [22 23]);
% make the encoding of the value in the population and add noise
for i=1:neurons_complete_z
    % scale the firing rate to proper values and compute fi
    z_population(i).ri = gauss_val(encoded_val_z, ...
                                   z_population(i).vi, ...
                                   sigma_z, ...
                                   scaling_factor) + ...
                                   etaz(i);
    % rate should be positive althought noise can make small vallues
    % negative
    z_population(i).ri = abs(z_population(i).ri);                               
end;
% plot the noisy hill of population activity encoding the given value
% index for neurons
j = 1;
for i=-z_pop_range:z_pop_range
    % display on even spacing of the entire input domain
    if(rem(i, z_spacing)==0)
        plot(i, z_population(j).ri, 'o');
        hold all;
        j = j+1;
    end;
end;
grid off;
set(gca, 'Box', 'off');     
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_z));
ylabel('Activity (spk/s)');
xlabel('Preferred value');

%% NETWORK DYNAMICS 
% define a 2D intermediate network layer on which the populations project
% assuming we are projecting the populations x and y and population z will
% encode an arbitrary function phi: z = phi(x, y)

%% FEEDFORWARD NETWORK CONNECTIVITY FROM INPUTS TO INTERMEDIATE LAYER
% connectivity matrix initialization
omega_h = 1;
omega_l = 0;
sum_rx = 0.0;
sum_ry = 0.0;

% connectivity matrix random initialization
J =  omega_l + ((omega_h - omega_l)/100).*rand(neurons_complete_x, neurons_complete_y);
% connectivity matrix type {smooth, sharp}
J_type = 'smooth';

% sharply peaked of J at i=j
for i=1:neurons_complete_x
    for j=1:neurons_complete_y
        % switch profile of weight matrix such that a more smoother
        switch(J_type)
            case 'smooth'
                % projection in the intermediate layer is obtained - Gauss
                J(i,j) = exp(-((i-j))^2/(2*(neurons_complete_x/10)*(neurons_complete_x/10)));
            case 'sharp'
                % for linear (sharp) profile of the weight matrix the
                % projection in the intermediate layer is noisier
                if(i==j)
                    J(i,j) = 1;
                end
        end
        end
    end
    
% stores the summed input activity for each neuron before intermediate
% layer
sum_hist_rx = zeros(1, neurons_complete_x);
sum_hist_ry = zeros(1, neurons_complete_y);

% stores the activity of each neuron in the intermediate layer as a
% superposition of the activities in the input layers
rxy_hist = zeros(neurons_complete_x, neurons_complete_y);

% compute the total input for each neuron in the intermediate layers
for i=1:neurons_complete_x
    for j=1:neurons_complete_y
        % reinit sum for every neuron in the intermediate layer
        sum_rx = 0.0;
        sum_ry = 0.0;

        % each input population contribution 
        for k = 1:neurons_complete_x
            sum_rx = sum_rx + J(i,k)*x_population(k).ri;
        end
        for l = 1:neurons_complete_y
            sum_ry = sum_ry + J(j,l)*y_population(l).ri;
        end
        % update history of activities
        sum_hist_rx(i,j) = sum_rx;
        sum_hist_ry(i,j) = sum_ry;
        % superimpose contributions from both populations 
        rxy = sum_rx + sum_ry;
        % update history for the intermediate layer neurons
        rxy_hist(i,j) = rxy;
    end
end

% normalized activity vector for neurons in the intermediate layer
rxy_normed = zeros(neurons_complete_x, neurons_complete_y);
% final activity of a neuron in the intermediate layer 
rij = zeros(neurons_complete_x, neurons_complete_y);

% choose the type of activation function for the projection step
% for the learning phase we might use a different activation function 
% activation_type = {linear, sigmoid}
activation_type = 'linear';

% assemble the intermediate layer and fill in with activity values
for i  = 1:neurons_complete_x
    for j = 1:neurons_complete_y
        % normalize the total activity such that we have consistent 
        % rate in the intermediate layer bound to [bkg_firing, max_firing]
        rxy_normed(i,j) = bkg_firing + ...
            ((rxy_hist(i,j) - min(rxy_hist(:)))*...
            (max_firing - bkg_firing))/...
            (max(rxy_hist(:) - min(rxy_hist(:))));
        switch(activation_type)
            case 'sigmoid'
                % compute the activation for each neuron - sigmoid activation 
                rij(i,j) = sigmoid(max_firing, ...
                                   max_firing/sigmoid_polarity_switch, ...
                                   rxy_normed(i,j));
            case 'linear'
                % compute the activation for each neuron - linear activation
                rij(i,j) = rxy_normed(i,j);
        end
        % build up the intermediate projection layer (no recurrency)
        projection_layer(i,j) = struct('i', i, ...
                                       'j', j, ...
                                       'rij', rij(i,j));
    end
end

%% VISUALIZATION OF INTERMEDIATE LAYER ACTIVITY (ONLY FEEDFORWARD PROP.)
%intermediate layer activity
% intialize the projected activity on the intermediate layer in aux var
% just for visualization 
projected_activity = zeros(neurons_complete_x, neurons_complete_y);
subplot(6, 4, [10 14]);
for i=1:neurons_complete_x
    for j=1:neurons_complete_y
        projected_activity(i,j) = projection_layer(i,j).rij;
    end
end
mesh([x_population.vi], [y_population.vi], projected_activity);
grid off;
set(gca, 'Box', 'off');  

%% RECURRENT CONNECTIVITY IN THE INTERMEDIATE LAYER

% get rid of the ridges in the activity profile of the intermediate layer
% keep only the bump of activity (Mexican hat - Difference Of Gaussians)

% parameters that control the shape of the W connectivity matrix in the
% intermediate layer 
We      = 1.5; % short range excitation strength We > Wi
Wi      = 0.5; % long range inhibition strength 
sigma_e = 10;  % excitation Gaussian profile sigma_e < sigma_i
sigma_i = 20;  % inhibiton Gaussian profile
W = zeros(neurons_complete_x, neurons_complete_y);

% build the recurrent connectivity matrix
for i=1:neurons_complete_x
    for j=1:neurons_complete_y
        for k = 1:neurons_complete_x-i
            for l = 1:neurons_complete_y-j
                W(i,j) = We*(exp(-((i-k)^2+(j-l)^2)/(2*sigma_e^2))) - ...
                         Wi*(exp(-((i-k)^2+(j-l)^2)/(2*sigma_i^2)));
            end
        end
    end
end

% stores the summed input activity for recurrent connections in intermed.
% layer
sum_hist_recurrent = zeros(neurons_complete_x, neurons_complete_y);
% stores the (un-normalized) activity of a neuron in the intermed. layer
interm_activities = zeros(neurons_complete_x, neurons_complete_y);

% build up the recurrent conectivity matrix in the intermediate layer so
% that the ridges are eliminated from the intermediate layer and only the
% bump persists

% using recurrency we have to introduce a dynamics (line attractor)
convergence_steps = 1;
% dynamics of the relaxation in the intermediate layer
rij_dot = zeros(neurons_complete_x, neurons_complete_y);
rij_dot_ant = 0;

rij_final = zeros(neurons_complete_x, neurons_complete_y, convergence_steps);
rij_final_ant = 0;
projection_layer_complete_ant = 0;

for t = 1:convergence_steps  
% loop through the projection layer (intermediate layer)    
for i=1:neurons_complete_x
    for j=1:neurons_complete_y
        % reinit sum for every neuron in the intermediate layer
        sum_rx = 0.0;
        sum_ry = 0.0;
        % recurrent connection contribution
        sum_recurrent = 0.0;
        
        % each input population contribution
        for k = 1:neurons_complete_x
            sum_rx = sum_rx + J(i,k)*x_population(k).ri;
        end
        for l = 1:neurons_complete_y
            sum_ry = sum_ry + J(j,l)*y_population(l).ri;
        end
        
        % fill in the global activity matrix
        interm_activities(i,j) = sum_rx + sum_ry;
        
        % recurrent connectivity contribution
        for k = 1:neurons_complete_x
            for l = 1:neurons_complete_y
                if(k~=i && l~=j)
                    sum_recurrent = sum_recurrent + W(i,j)*...
                                                    rxy_hist(k,l);
                end
            end
        end
        
        % update history of activities
        sum_hist_rx(i,j) = sum_rx;
        sum_hist_ry(i,j) = sum_ry;
        sum_hist_recurrent(i,j) = sum_recurrent;
        
        % superimpose contributions from both populations and reccurency
        rxy = interm_activities(i,j) + sum_hist_recurrent(i,j);
        
        % update history for the intermediate layer neurons
        rxy_hist(i,j) = rxy;
        
        % normalize the total activity such that we have consistent
        % rate in the intermediate layer bound to [bkg_firing, max_firing]
        rxy_normed(i,j) = bkg_firing + ...
            ((rxy_hist(i,j) - min(rxy_hist(:)))*...
            (max_firing - bkg_firing))/...
            (max(rxy_hist(:) - min(rxy_hist(:))));
        switch(activation_type)
            case 'sigmoid'
                % compute the activation for each neuron - sigmoid activation
                rij(i,j) = sigmoid(max_firing, ...
                                   max_firing/sigmoid_polarity_switch, ...
                                   rxy_normed(i,j));
            case 'linear'
                % compute the activation for each neuron - linear activation
                rij(i,j) = rxy_normed(i,j);
        end
        % build up the intermediate projection layer (including recurrency)
        projection_layer_complete(i,j) = struct('i', i, ...
                                       'j', j, ...
                                       'rij', rij(i,j));
                                   
        % compute the change in activity
        rij_dot(i,j) = projection_layer_complete(i,j).rij;
        rij_final(i,j,t) = rij_final_ant + (0.1*(rij_dot(i,j) + rij_dot_ant)*.5);
        % update history 
        rij_dot_ant = rij_dot(i,j);
        rij_final_ant = rij_final(i,j,t); 
    end
end
end % convergence steps

close all;
mesh(1:neurons_complete_x, 1:neurons_complete_y, sum_hist_recurrent(1:neurons_complete_x, 1:neurons_complete_y))
return 

%% FEEDFORWARD CONNECTIVITY FROM INTERMEDIATE LAYER TO OUTPUT POPULATION
% after relaxation the intermediate layer activity is projected onto the
% output population



%% VISUALIZATION OF INTERMEDIATE LAYER ACTIVITY (AFTER DYNAMICS)
% intermediate layer activity after net dynamics relaxed
projected_activity = zeros(neurons_complete_x, neurons_complete_y);

subplot(6, 4, [11 15]);
for i=1:neurons_complete_x
    for j=1:neurons_complete_y
        projected_activity(i,j) = rij_final(i,j,convergence_steps);
    end
end
mesh([x_population.vi], [y_population.vi], projected_activity);
grid off;
set(gca, 'Box', 'off');  