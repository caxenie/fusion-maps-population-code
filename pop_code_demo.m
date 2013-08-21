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
clc; clf;
close all;

% define the 1D populations (symmetric (size wise) populations)
neurons_num_x = 41;  % number of neurons in the input populations
neurons_num_y = 41;
neurons_num_z = 41;
noise_scale = 10;

% neuron information (neuron index i)
%   i       - index in the population 
%   ri      - activity of the neuron (firing rate)
%   fi      - tuning curve (e.g. Gaussian)
%   vi      - preferred value - even distribution in the range
%   etai    - noise value - zero mean and correlated

% scaling factor for tuning curves
bkg_firing = 10; % spk/s - background firing rate
scaling_factor = 80; % motivated by the typical background and upper spiking rates

%% generate first population and initialize
% preallocate
vi=[];
fi=[];
ri=[];
% zero mean noise 
etax = randn(neurons_num_x, 1)*noise_scale;
% population standard deviation - coarse (big val) / sharp receptive field
sigma_x = 10;
% population range of values (+/-)
x_pop_range = 100;
% peak to peak spacing in tuning curves
x_spacing = x_pop_range/((neurons_num_x-1)/2);
% init population
for i=1:neurons_num_x
    % evenly distributed preferred values in the interval
    vi(i) = -x_pop_range+(i-1)*(x_pop_range/((neurons_num_x-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(vi(i), ...
                               sigma_x, ...
                               x_pop_range, ...
                               scaling_factor);
    fi(i).p = pts;
    fi(i).v = vals;
    x_population(i) = struct('i', i, ...
                            'vi', vi(i),...
                            'fi', fi(i), ...
                            'etai', etax(i),...
                            'ri', randi([bkg_firing , scaling_factor]));
end;

%% generate second population and initialize
% preallocate
vi=[];
fi=[];
ri=[];
% zero mean noise
etay = randn(neurons_num_y, 1)*noise_scale;
% population standard deviation - coarse (big val) / sharp receptive field
sigma_y = 17;
% population range of values (+/-)
y_pop_range = 100;
% peak to peak spacing in tuning curves
y_spacing = y_pop_range/((neurons_num_y-1)/2);
% init population
for i=1:neurons_num_y
    % evenly distributed preferred values in the interval
    vi(i) = -y_pop_range+(i-1)*(y_pop_range/((neurons_num_y-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(vi(i), ...
                               sigma_y, ...
                               y_pop_range, ...
                               scaling_factor);
    fi(i).p = pts;
    fi(i).v = vals;
    y_population(i) = struct('i', i, ...
                            'vi', vi(i),...
                            'fi', fi(i), ...
                            'etai', etay(i),...
                            'ri', randi([bkg_firing , scaling_factor]));
end;

%% generate third population and initialize
% preallocate
vi=[];
fi=[];
ri=[];
% zero mean noise
etaz = randn(neurons_num_z, 1)*noise_scale;
% population standard deviation - coarse (big val) / sharp receptive field
sigma_z = 23;
% population range of values (+/-) 
z_pop_range = 200;
% peak to peak spacing in tuning curves
z_spacing = z_pop_range/((neurons_num_z-1)/2);
% init population
for i=1:neurons_num_z
    % evenly distributed preferred values in the interval
    vi(i) = -z_pop_range+(i-1)*(z_pop_range/((neurons_num_z-1)/2));
    % tuning curve of the neuron
    [pts, vals] = gauss_tuning(vi(i), ...
                               sigma_z, ...
                               z_pop_range, ...
                               scaling_factor);
    fi(i).p = pts;
    fi(i).v = vals;
    z_population(i) = struct('i', i, ...
                            'vi', vi(i),...
                            'fi', fi(i), ...
                            'etai', etaz(i),...
                            'ri', randi([bkg_firing , scaling_factor]));
end;

%% NETWORK DYNAMICS 
% define a 2D intermediate network layer on which the populations project
% assuming we are projecting the populations x and y and population z will
% encode an arbitrary function phi: z = phi(x, y)

% connectivity matrix initialization
omega_h = 1;
omega_l = 0;
for i=1:neurons_num_x
    rx(i)=0.0;
end
for i=1:neurons_num_y
    ry(i)=0.0;
end
J = omega_l + (omega_h - omega_l).*rand(neurons_num_x, neurons_num_y);
% compute the individual inputs for each neuron in the intermediate layer
for i=1:neurons_num_x
    for j=1:neurons_num_y
        for t = 1:neurons_num_x
            rx(i) = rx(i) + J(i, t)*x_population(t).ri;
        end
        for z = 1:neurons_num_y
            ry(j) = ry(j) + J(j, z)*y_population(z).ri;
        end
        rxy(i,j) = rx(i)+ry(j);
    end
end

for i=1:neurons_num_x
    for j=1:neurons_num_y
        % build up the intermediate projection layer
        % compute the activation for each neuron 
        rij(i,j) = sigmoid(rxy(i,j));
        projection_layer(i,j) = struct('i', i, ...
                                       'j', j, ...
                                       'rij', rij(i,j));
    end;
end;

%% VISUALIZATION
figure;
%% first population 
% plot the tunning curves of all neurons for the first population
subplot(6, 3, 1);
for i=1:neurons_num_x
    plot(x_population(i).fi.p, x_population(i).fi.v);
    hold all;
end;
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');
xlabel('Preferred value');

% plot the encoded value in the population
subplot(6, 3, 4);
encoded_val_x = -23;
% make the encoding of the value in the population and add noise
for i=1:neurons_num_x
    % scale the firing rate to proper values and compute fi
    x_population(i).ri = gauss_val(encoded_val_x, ...
                                   x_population(i).vi, ...
                                   sigma_x, ...
                                   scaling_factor) + ...
                                   etax(i);
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
grid on;       
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_x));
ylabel('Activity (spk/s)');
xlabel('Preferred value');

%% second population 
subplot(6, 3, 3);
for i=1:neurons_num_y
    plot(y_population(i).fi.p, y_population(i).fi.v);
    hold all;
end;
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');
xlabel('Preferred value');

% plot the encoded value in the population
subplot(6, 3, 6);
encoded_val_y = 18;
% make the encoding of the value in the population and add noise
for i=1:neurons_num_y
    % scale the firing rate to proper values and compute fi
    y_population(i).ri = gauss_val(encoded_val_y, ...
                                   y_population(i).vi, ...
                                   sigma_y, ...
                                   scaling_factor) + ...
                                   etay(i);
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
grid on;       
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_y));
ylabel('Activity (spk/s)');
xlabel('Preferred value');


%% third population 
subplot(6, 3, 14);
for i=1:neurons_num_z
    plot(z_population(i).fi.p, z_population(i).fi.v);
    hold all;
end;
title('Tuning curves of the neural population');
ylabel('Activity (spk/s)');
xlabel('Preferred value');

% plot the encoded value in the population
subplot(6, 3, 17);
encoded_val_z = 75;
% make the encoding of the value in the population and add noise
for i=1:neurons_num_z
    % scale the firing rate to proper values and compute fi
    z_population(i).ri = gauss_val(encoded_val_z, ...
                                   z_population(i).vi, ...
                                   sigma_z, ...
                                   scaling_factor) + ...
                                   etaz(i);
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
grid on;       
title(sprintf('Noisy activity of the population encoding the value %d', encoded_val_z));
ylabel('Activity (spk/s)');
xlabel('Preferred value');
