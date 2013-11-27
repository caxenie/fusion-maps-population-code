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
noise_scale = 10;

% neuron information (neuron index i)
%   i       - index in the population 
%   ri      - activity of the neuron (firing rate)
%   fi      - tuning curve (e.g. Gaussian)
%   vi      - preferred value - even distribution in the range
%   etai    - neuronal noise value - zero mean and tipically correlated

% scaling factor for tuning curves
bkg_firing = 10; % spk/s - background firing rate
max_firing = 100;% spk/s - maximum firing rate

% ========================================================================
% demo params 
encoded_val_x = -30;
encoded_val_y = -10;
% the output should be computed depending on the embedded function phi 
% this is just initialization 
encoded_val_z = 0 ;
% ========================================================================

% choose the type of activation function for the projection step
% for the learning phase we might use a different activation function 
% activation_type = {linear, sigmoid}
projection_activation_type = 'linear';
learning_activation_type = 'sigmoid';

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
for idx=1:neurons_complete_x
    % evenly distributed preferred values in the interval
    vix(idx) = -x_pop_range+(idx-1)*(x_pop_range/((neurons_complete_x-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(vix(idx), ...
                               sigma_x, ...
                               x_pop_range, ...
                               max_firing);
    fix(idx).p = pts;
    fix(idx).v = vals;
    x_population(idx) = struct('i',   idx, ...
                            'vi',   vix(idx),...
                            'fi',   fix(idx), ...
                            'etai', etax(idx),...
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
for idx=1:neurons_complete_y
    % evenly distributed preferred values in the interval
    viy(idx) = -y_pop_range+(idx-1)*(y_pop_range/((neurons_complete_y-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(viy(idx), ...
                               sigma_y, ...
                               y_pop_range, ...
                               max_firing);
    fiy(idx).p = pts;
    fiy(idx).v = vals;
    y_population(idx) = struct('i',   idx, ...
                            'vi',   viy(idx),...
                            'fi',   fiy(idx), ...
                            'etai', etay(idx),...
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
z_pop_range = 100;
% peak to peak spacing in tuning curves
z_spacing = z_pop_range/((neurons_complete_z-1)/2);
% init population
for idx=1:neurons_complete_z
    % evenly distributed preferred values in the interval
    viz(idx) = -z_pop_range+(idx-1)*(z_pop_range/((neurons_complete_z-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(viz(idx), ...
                               sigma_z, ...
                               z_pop_range, ...
                               max_firing);
    fiz(idx).p = pts;
    fiz(idx).v = vals;
    z_population(idx) = struct('i',   idx, ...
                            'vi',   viz(idx),...
                            'fi',   fiz(idx), ...
                            'etai', etaz(idx),...
                            'ri',   randi([bkg_firing , max_firing]));
end;
%% VISUALIZATION
figure;
set(gcf,'color','w');

% first population 
% plot the tunning curves of all neurons for the first population
subplot(6, 4, [1 2]);
for idx=1:neurons_complete_x
    plot(x_population(idx).fi.p, x_population(idx).fi.v);
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

for idx=1:neurons_complete_x
    % scale the firing rate to proper values and compute fi
    x_population(idx).ri = gauss_val(encoded_val_x, ...
                                   x_population(idx).vi, ...
                                   sigma_x, ...
                                   max_firing) + ...
                                   etax(idx);
    % rate should be positive althought noise can make small values
    % negative
    x_population(idx).ri = abs(x_population(idx).ri);
end;
% plot the noisy hill of population activity encoding the given value
% index for neurons
jdx = 1;
for idx=-x_pop_range:x_pop_range
    % display on even spacing of the entire input domain
    if(rem(idx, x_spacing)==0)
        plot(idx, x_population(jdx).ri, 'o');
        hold all;
        jdx = jdx+1;
    end;
end; 
grid off;
set(gca, 'Box', 'off');
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_x));
ylabel('Activity (spk/s)');
xlabel('Preferred value');

% second population 
subplot(6, 4, [3 4]);
for idx=1:neurons_complete_y
    plot(y_population(idx).fi.p, y_population(idx).fi.v);
    hold all;
end;
grid off;
set(gca, 'Box', 'off');
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');

% plot the encoded value in the population
subplot(6, 4, [7 8]);
% make the encoding of the value in the population and add noise
for idx=1:neurons_complete_y
    % scale the firing rate to proper values and compute fi
    y_population(idx).ri = gauss_val(encoded_val_y, ...
                                   y_population(idx).vi, ...
                                   sigma_y, ...
                                   max_firing) + ...
                                   etay(idx);
    % rate should be positive althought noise can make small vallues
    % negative
    y_population(idx).ri = abs(y_population(idx).ri);                               
end;
% plot the noisy hill of population activity encoding the given value
% index for neurons
jdx = 1;
for idx=-y_pop_range:y_pop_range
    % display on even spacing of the entire input domain
    if(rem(idx, y_spacing)==0)
        plot(idx, y_population(jdx).ri, 'o');
        hold all;
        jdx = jdx+1;
    end;
end;
grid off;
set(gca, 'Box', 'off');      
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_y));
ylabel('Activity (spk/s)');
xlabel('Preferred value');

% third population 
subplot(6, 4, [18 19]);
for idx=1:neurons_complete_z
    plot(z_population(idx).fi.p, z_population(idx).fi.v);
    hold all;
end;
grid off;
set(gca, 'Box', 'off');
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');

% plot the encoded value in the population
subplot(6, 4, [21 22]);
% make the encoding of the value in the population and add noise
for idx=1:neurons_complete_z
    % output population initialized randomly with a value the final encoded 
    % value will be determined by the function embedded in phi
    z_population(idx).ri = gauss_val(encoded_val_z, ...
                                   z_population(idx).vi, ...
                                   sigma_z, ...
                                   max_firing) + ...
                                   etaz(idx);
    % rate should be positive althought noise can make small vallues
    % negative
    z_population(idx).ri = abs(z_population(idx).ri);                               
end;
% plot the noisy hill of population activity encoding the given value
% index for neurons
jdx = 1;
for idx=-z_pop_range:z_pop_range
    % display on even spacing of the entire input domain
    if(rem(idx, z_spacing)==0)
        plot(idx, z_population(jdx).ri, 'o');
        hold all;
        jdx = jdx+1;
    end;
end;
grid off;
set(gca, 'Box', 'off');     
title('Noisy intialization activity of the population with a random value (t --> 0)');
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
for idx=1:neurons_complete_x
    for jdx=1:neurons_complete_y
        % switch profile of weight matrix such that a more smoother
        switch(J_type)
            case 'smooth'
                % projection in the intermediate layer is obtained - Gauss
                J(idx,jdx) = exp(-((idx-jdx))^2/(2*(neurons_complete_x/10)*(neurons_complete_x/10)));
            case 'sharp'
                % for linear (sharp) profile of the weight matrix the
                % projection in the intermediate layer is noisier
                if(idx==jdx)
                    J(idx,jdx) = 1;
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
for idx=1:neurons_complete_x
    for jdx=1:neurons_complete_y
        % reinit sum for every neuron in the intermediate layer
        sum_rx = 0.0;
        sum_ry = 0.0;

        % each input population contribution 
        for k = 1:neurons_complete_x
            sum_rx = sum_rx + J(idx,k)*x_population(k).ri;
        end
        for l = 1:neurons_complete_y
            sum_ry = sum_ry + J(jdx,l)*y_population(l).ri;
        end
        % update history of activities
        sum_hist_rx(idx,jdx) = sum_rx;
        sum_hist_ry(idx,jdx) = sum_ry;
        % superimpose contributions from both populations 
        rxy = sum_rx + sum_ry;
        % update history for the intermediate layer neurons
        rxy_hist(idx,jdx) = rxy;
    end
end

% normalized activity vector for neurons in the intermediate layer
rxy_normed = zeros(neurons_complete_x, neurons_complete_y);
% final activity of a neuron in the intermediate layer 
rij = zeros(neurons_complete_x, neurons_complete_y);

% assemble the intermediate layer and fill in with activity values
for idx  = 1:neurons_complete_x
    for jdx = 1:neurons_complete_y
        % normalize the total activity such that we have consistent 
        % rate in the intermediate layer bound to [bkg_firing, max_firing]
        rxy_normed(idx,jdx) = bkg_firing + ...
            ((rxy_hist(idx,jdx) - min(rxy_hist(:)))*...
              (max_firing - bkg_firing))/...
            (max(rxy_hist(:) - min(rxy_hist(:))));
        switch(projection_activation_type)
            case 'sigmoid'
                % compute the activation for each neuron - sigmoid activation 
                rij(idx,jdx) = sigmoid(max_firing, ...
                                       max_firing, ...
                                       rxy_normed(idx,jdx));
            case 'linear'
                % compute the activation for each neuron - linear activation
                rij(idx,jdx) = rxy_normed(idx,jdx);
        end
        % build up the intermediate projection layer (no recurrency)
        projection_layer(idx,jdx) = struct('i', idx, ...
                                       'j', jdx, ...
                                       'rij', rij(idx,jdx));
    end
end

%% VISUALIZATION OF INTERMEDIATE LAYER ACTIVITY (ONLY FEEDFORWARD PROP.)
%intermediate layer activity
% intialize the projected activity on the intermediate layer in aux var
% just for visualization 
projected_activity = zeros(neurons_complete_x, neurons_complete_y);
h(1) = subplot(6, 4, [10 14]);
for idx=1:neurons_complete_x
    for jdx=1:neurons_complete_y
        projected_activity(idx,jdx) = projection_layer(idx,jdx).rij;
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
We      = 10.0;     % short range excitation strength We > Wi
Wi      = 0.5;      % long range inhibition strength 
sigma_e = 7;        % excitation Gaussian profile sigma_e < sigma_i
sigma_i = 14;       % inhibiton Gaussian profile
W = zeros(neurons_complete_x, neurons_complete_y, neurons_complete_x, neurons_complete_y);

% build the recurrent connectivity matrix
for idx=1:neurons_complete_x
    for jdx = 1:neurons_complete_y
        for k = 1:neurons_complete_x
            for l = 1:neurons_complete_y
                W(idx,jdx,k,l) = We*(exp(-((idx-k)^2+(jdx-l)^2)/(2*sigma_e^2))) - ...
                                 Wi*(exp(-((idx-k)^2+(jdx-l)^2)/(2*sigma_i^2)));
            end
        end
    end
end

% show animated movement of the mexican hat 
% for i = 1:neurons_complete_x
%     for j = 1:neurons_complete_y
%         surf(W(1:neurons_complete_x, 1:neurons_complete_y, i, j));
%         pause(0.1);
%     end
% end

% stores the summed input activity for recurrent connections in intermed.
% layer
sum_hist_recurrent = zeros(neurons_complete_x, neurons_complete_y);
% stores the (un-normalized) activity of a neuron in the intermed. layer
interm_activities = zeros(neurons_complete_x, neurons_complete_y);

% build up the recurrent conectivity matrix in the intermediate layer so
% that the ridges are eliminated from the intermediate layer and only the
% bump persists

% using recurrency we have to introduce a dynamics (line attractor)
convergence_steps = 5;
% dynamics of the relaxation in the intermediate layer
rij_dot = zeros(neurons_complete_x, neurons_complete_y);
rij_dot_ant = 0;

rij_final = zeros(neurons_complete_x, neurons_complete_y, convergence_steps);
rij_final_ant = 0;
projection_layer_complete_ant = 0;

for t = 1:convergence_steps  
% loop through the projection layer (intermediate layer)    
for idx=1:neurons_complete_x
    for jdx=1:neurons_complete_y
        % reinit sum for every neuron in the intermediate layer
        sum_rx = 0.0;
        sum_ry = 0.0;
        % recurrent connection contribution
        sum_recurrent = 0.0;
        
        % each input population contribution
        for k = 1:neurons_complete_x
            sum_rx = sum_rx + J(idx,k)*x_population(k).ri;
        end
        for l = 1:neurons_complete_y
            sum_ry = sum_ry + J(jdx,l)*y_population(l).ri;
        end
        
        % fill in the global activity matrix
        interm_activities(idx,jdx) = sum_rx + sum_ry;
        
        % recurrent connectivity contribution
        for k = 1:neurons_complete_x
            for l = 1:neurons_complete_y
                if(k~=idx && l~=jdx)
                    sum_recurrent = sum_recurrent + W(idx,jdx, k, l)*...
                                                    interm_activities(k,l);
                end
            end
        end
        
        % update history of activities
        sum_hist_rx(idx,jdx) = sum_rx;
        sum_hist_ry(idx,jdx) = sum_ry;
        sum_hist_recurrent(idx,jdx) = sum_recurrent;
        
        % superimpose contributions from both populations and reccurency
        rxy = interm_activities(idx,jdx) + sum_hist_recurrent(idx,jdx);
        
        % update history for the intermediate layer neurons
        rxy_hist(idx,jdx) = rxy;
        
        % normalize the total activity such that we have consistent
        % rate in the intermediate layer bound to [bkg_firing, max_firing]
        rxy_normed(idx,jdx) = bkg_firing + ...
            ((rxy_hist(idx,jdx) - min(rxy_hist(:)))*...
            (max_firing - bkg_firing))/...
            (max(rxy_hist(:) - min(rxy_hist(:))));
        switch(learning_activation_type)
            case 'sigmoid'
                % compute the activation for each neuron - sigmoid activation
                rij(idx,jdx) = sigmoid(max_firing, ...
                                       max_firing, ...
                                       rxy_normed(idx,jdx));
            case 'linear'
                % compute the activation for each neuron - linear activation
                rij(idx,jdx) = rxy_normed(idx,jdx);
        end
        % build up the intermediate projection layer (including recurrency)
        projection_layer_complete(idx,jdx) = struct('i', idx, ...
                                                    'j', jdx, ...
                                                    'rij', rij(idx,jdx));
                                   
        % compute the change in activity
        rij_dot(idx,jdx) = projection_layer_complete(idx,jdx).rij - projection_layer_complete_ant;
        % integrate activity in time (t->Inf)
        rij_final(idx,jdx,t) = rij_final_ant + ((rij_dot(idx,jdx) + rij_dot_ant)*.5);
        % update history 
        rij_dot_ant = rij_dot(idx,jdx);
        rij_final_ant = rij_final(idx,jdx,t); 
        projection_layer_complete_ant = projection_layer_complete(idx,jdx).rij;
    end
end
end % convergence steps

%% FEEDFORWARD CONNECTIVITY FROM INTERMEDIATE LAYER TO OUTPUT POPULATION
% after relaxation the intermediate layer activity is projected onto the
% output population after relaxation (t->Inf)

% sum of activity from intermediate layer to output layer
sum_interm_out = 0; 
for idx=1:neurons_complete_z
    sum_interm_out = 0;
    for jdx= 1:neurons_complete_x
        for k = 1:neurons_complete_y
            sum_interm_out = sum_interm_out + ...
                             G_map(z_population(idx).vi - phi(x_population(jdx).vi, y_population(k).vi))*...
                             (rij_final(jdx, k, convergence_steps));
        end
    end
    z_population(idx).ri = sum_interm_out;
end

% normalize the activity in the output population
z_population_normed = zeros(1, neurons_complete_z);
min_zpop = min([z_population(:).ri]);
max_zpop = max([z_population(:).ri]);
for idx = 1:neurons_complete_z
    z_population_normed(idx) = bkg_firing + ...
            ((z_population(idx).ri - min_zpop)*...
            (max_firing - bkg_firing))/...
            (max_zpop - min_zpop);
end
       
for idx = 1:neurons_complete_z
    z_population(idx).ri  = z_population_normed(idx);
end

%% VISUALIZATION OF INTERMEDIATE LAYER ACTIVITY (AFTER DYNAMICS)
% intermediate layer activity after net dynamics relaxed
projected_activity = zeros(neurons_complete_x, neurons_complete_y);

h(2) = subplot(6, 4, [11 15]);
for idx=1:neurons_complete_x
    for jdx=1:neurons_complete_y
        projected_activity(idx,jdx) = projection_layer_complete(idx,jdx).rij; % final value after relaxation
    end
end
mesh([x_population.vi], [y_population.vi], projected_activity);
grid off;
set(gca, 'Box', 'off');  

% link axes for the 2 presentations of the intermediate layer activity
linkprop([h(1) h(2)], 'CameraPosition');

%plot the encoded value in the output population after the network relaxed
%and the values are settles
subplot(6, 4, [23 24]);
% plot the noisy hill of population activity 
jdx = 1;
for idx=-z_pop_range:z_pop_range
    % display on even spacing of the entire input domain
    if(rem(idx, z_spacing)==0)
        plot(idx, z_population(jdx).ri, 'o');
        hold all;
        jdx = jdx+1;
    end;
end;
% the encoded value in the output population is given by the embedded
% function phi
grid off;
set(gca, 'Box', 'off');     
title(sprintf('Noisy activity of the population encoding the value %d after relaxation (t --> Inf)', phi(encoded_val_x, encoded_val_y)));
ylabel('Activity (spk/s)');
xlabel('Preferred value');