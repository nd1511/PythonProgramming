%   Prepare the CNN inputs and 2 more vectors for cepstral domain         %
%   approach III.                                                         %
%   Input: 
%         1- Mean and variance of training data:                          %
%            mean_std_of_TrainData_g711_best.mat                          %
%         2- Coded speech for test: exapmle_s_g711_coded.raw              %
%   Output: 
%         1- CNN input vector: type_3_cnn_input_ceps_v73.mat              %
%         2- Residual cepstral coefficients vector: type_3_ceps_resi.mat  %
%         3- Phase angel vector: type_3_pha_ang.mat                       %



clear; clc; 
addpath(genpath(pwd));
%   1- Parameters setting;

% Framestructure parameters
frameLen = 0.032 * 8000; % 32ms @ NB
leng_step = 0.010 * 8000; % 10ms @ NB
frameLen_process = 0.020 * 8000; % 20ms  @ NB
wind = hann(frameLen_process,'periodic'); % window length = processing frame length
K_fft = frameLen * 2; % NOT for prime Types
% Load mean and std values from training
mean_std_file = './data/mean_std_of_TrainData_g711_best.mat';

%  2- Coded speech loading, cepstral domain, and CNN input prepare
% - Load coded speech 
legacy_dec_out = loadshort('./dataset/exapmle_s_g711_coded.raw');
s_leng = length(legacy_dec_out);
legacy_dec_out = (legacy_dec_out./2^15); % make signal within[-1,+1]

% - Framing 
% Given that the received speech is 10ms/frame
vor_zero_num = 0.010 * 8000; % 5ms for pre-zero-padding
s_leng_vor = vor_zero_num + s_leng;
% Check whether need to zero-padding at the end
mod_num = mod(s_leng_vor, frameLen_process);
if mod_num == 0,
    nach_zero_num = 0;
else
    nach_zero_num = frameLen_process - mod_num;
end
s_leng_vor_nach = s_leng_vor + nach_zero_num;
dec_out_zero_pad = [zeros(vor_zero_num,1) ; legacy_dec_out ;...
    zeros(nach_zero_num,1)];
num_frame = (s_leng_vor_nach - frameLen_process)/leng_step + 1;

% Processing frame-wise
for k = 1 : num_frame,
    s_frame = dec_out_zero_pad( (k-1)*leng_step+1 : frameLen_process+(k-1)*leng_step );

    % - Cepstral domain transformation
    [~,ceps_env,ceps_resi_temp,ang_s,~,~] = sub_func_time_fram2ceps_coeff_pha(s_frame,wind,K_fft);
    s_coeff = ceps_env;
    ceps_resi(k,:) = ceps_resi_temp;
    inputTestSet(k,:) = s_coeff.';
    ang_mat(k,:) = ang_s.';
    % Display processing percentage!
    if mod(k,floor(num_frame/10)) == 0,
        disp([num2str(k) ' frames out of ' num2str(num_frame) ' frames finish prepared!']);
        disp([num2str(k/num_frame*100) '% finished!']);
    end
end

% - CNN input preparation
% normalization to unit mean, 0 variance
load(mean_std_file); 
inputTestNorm = zeros(size(inputTestSet));
for k=1:size(inputTestSet,1)
    inputTestNorm(k,:) = (inputTestSet(k,:) - mean_of_every_dim)./std_of_every_dim;
end
% - CNN input and two more vectors need to be stored
save('./data/type_3_cnn_input_ceps_v73.mat','inputTestNorm','-v7.3'); 
save('./data/type_3_ceps_resi.mat','ceps_resi');
save('./data/type_3_pha_ang.mat','ang_mat');




