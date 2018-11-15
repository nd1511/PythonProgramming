%   Prepare of training (or validation) data for CNN for cepstral         %
%   domain approach III.                                                  %
%                                                                         %
%   Input: 
%         1- Uncoded speech for training: example_uncoded_train_s.raw     %
%         2- Uncoded speech for validation: example_uncoded_valid_s.raw   %
%         3- Coded speech for training: example_coded_train_s.raw         %
%         4- Coded speech for validation: example_coded_valid_s.raw       %
%   Output: 
%         1- Training input: Train_inputSet_g711.mat                      %
%         2- Training target: Train_targetSet_g711.mat                    %
%         3- Validation input: Validation_inputSet_g711.mat               %
%         4- Validation target: Validation_targetSet_g711.mat             %
%         5- Mean and variance of training data:                          %
%            mean_std_of_TrainData_g711_example.mat                       %

clear; clc;
addpath(genpath(pwd));
data_type_str_vec = {'train_data', 'validation_data'};

for data_type_ind = 1 : 2,

    % - Produce training and validation data 
    data_type = data_type_str_vec{data_type_ind}; 

    % - Framestructure parameters (same as in Test)
    Fs = 8000;
    leng_step = 0.010 * Fs; 
    frameLen_process = 0.020 * Fs; 
    K_fft = (0.032 * Fs) * 2; 
    wind = hann(frameLen_process,'periodic');

    % - Load training and validation speech set (only example speech shown here)
    % For training (or validation) target
    if strcmp(data_type,'train_data'),
        speech = loadshort('./dataset/example_uncoded_train_s.raw'); 
    else
        speech = loadshort('./dataset/example_uncoded_valid_s.raw'); 
    end
    speech = (speech./2^15); % Convert to wav file
    speech = speech';
    s_leng = length(speech);

    % For training (or validation) input
    if strcmp(data_type,'train_data'),
        cod_speech = loadshort('./dataset/example_coded_train_s.raw'); 
    else
        cod_speech = loadshort('./dataset/example_coded_valid_s.raw'); 
    end
    cod_speech = (cod_speech./2^15); % Convert to wav file
    cod_speech = cod_speech';

    % VAD settings 
    vad_threrod = 0.0001; % VAD threshold 
    s_power = 1/s_leng * sum((speech).^2);
    num_vad_ind = 0;
    vad_ind = [];

    % Processing frame-wise
    num_frame = floor((s_leng-frameLen_process)/leng_step) + 1;
    inputSet = [];
    targetSet = [];
    for k = 1 : num_frame
        % Framing
        s_ind_vor  = (k-1)*leng_step + 1;
        s_ind_nach = (k-1)*leng_step + frameLen_process;
        s_frame = speech( s_ind_vor : s_ind_nach );
        s_hat_frame = cod_speech( s_ind_vor : s_ind_nach );
        s_frame_power(k) = 1/frameLen_process * sum((s_frame).^2);

        % VAD check
        if s_frame_power(k)/s_power > vad_threrod,
            num_vad_ind = num_vad_ind + 1;
            vad_ind(num_vad_ind) = k;
        end

        %  Cepstral coefficient
        [~,ceps_env,~,~,~,~] = sub_func_time_fram2ceps_coeff_pha(s_frame.',wind,K_fft);
        [~,ceps_env_hat,~,~,~] = sub_func_time_fram2ceps_coeff_pha(s_hat_frame.',wind,K_fft);

        framesOrg_spec(k,:) = ceps_env.';
        framesQua_spec(k,:) = ceps_env_hat.';
    end 

    % Save to data pairs
    inputSet = [inputSet; framesQua_spec(vad_ind, :)];
    targetSet = [targetSet; framesOrg_spec(vad_ind, :)];

    % Chech & delete NaN number
    [nan_chek_mat1] = isnan(inputSet);
    nan_chek_vec1 = sum(nan_chek_mat1,2);
    non_nan_ind1 = find(~nan_chek_vec1);
    [nan_chek_mat2] = isnan(targetSet);
    nan_chek_vec2 = sum(nan_chek_mat2,2);
    non_nan_ind2 = find(~nan_chek_vec2);

    non_nan_ind = intersect(non_nan_ind1,non_nan_ind2);
    inputSet  = inputSet(non_nan_ind,:);
    targetSet = targetSet(non_nan_ind,:);

    % - Normalization for unit mean, 0 variance
    mean_of_every_dim = mean(inputSet,1);
    % mean_of_every_dim = zeros(1,size(inputSet,2));
    std_of_every_dim = std(inputSet,[],1);
    inputSetNorm = zeros(size(inputSet));
    for k=1:size(inputSet,1)
        inputSetNorm(k,:) = (inputSet(k,:) - mean_of_every_dim)./std_of_every_dim;
    end

    % - Save data 
    if strcmp(data_type,'train_data'),
        save('./data/Train_inputSet_g711.mat','inputSetNorm');
        save('./data/Train_targetSet_g711.mat','targetSet');
        % Save mean and variance value on Training Set
        save('./data/mean_std_of_TrainData_g711_example.mat','mean_of_every_dim','std_of_every_dim');
    else
        save('./data/Validation_inputSet_g711.mat','inputSetNorm');
        save('./data/Validation_targetSet_g711.mat','targetSet');
    end

end
