% Demo software usign population code for estimating arbitrary functions
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

% general cleanup
clear all;
clc; clf;
close all;

% define the 1D populations (symmetric (size wise) populations)
neurons_num = 41;  % number of neurons in the input populations

% neuron information (neuron index i)
%   i       - index in the population 
%   ri      - activity of the neuron (firing rate)
%   fi      - tuning curve (e.g. Gaussian)
%   vi      - preferred value - even distribution in the range
%   etai    - noise value - zero mean and correlated

% preallocate
vi=[];
fi=[];
ri=[];

% generate first population and initialize
% zero mean noise 
etax = randn(neurons_num, 1);
% population standard deviation - coarse (big val) / sharp receptive field
sigma_x = 18;
% first population range of values (+/-)
x_pop_range = 100;
% scaling factor for tuning curves
bkg_firing = 10; % spk/s - background firing rate
scaling_factor = 80; % motivated by the typical background and upper spiking rates
% init population
for i=1:neurons_num
    % evenly distributed preferred values in the interval
    vi(i) = -x_pop_range+(i-1)*(x_pop_range/((neurons_num-1)/2));
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

% plot the tunning curves of all neurons 
figure(1);
for i=1:neurons_num
    plot(x_population(i).fi.p, x_population(i).fi.v);
    hold all;
end;

% plot the encoded value in the population
figure(2);
encoded_val = -23;
% make the encoding of the value in the population and add noise
for i=1:neurons_num
    % scale the firing rate to proper values and compute fi
    x_population(i).ri = gauss_val(encoded_val, ...
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
    if(rem(i,(abs(vi(1))-abs(vi(2))))==0)
        plot(i, x_population(j).ri, 'o');
        hold all;
        j = j+1;
    end;
end;





