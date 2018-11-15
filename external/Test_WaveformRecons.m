






%   Waveform reconstruction for cepstral domain approach with framework   %
%   structure III                                                         %
%   Input: 
%         1- CNN output vector: type_3_cnn_output_ceps.mat                %
%         2- Residual cepstral coefficients vector: type_3_ceps_resi.mat  %
%         3- Phase angel vector: type_3_pha_ang.mat                       %
%         4- Coded speech: exapmle_s_g711_coded.raw                       %
%   Output: 
%         1- Postprocessed waveform: cnn_postprocessed_8K_out.raw         %

clear;
clc;

addpath(genpath(pwd));

% - Framestructure parameters (same as in Input Prepare)
frameLen = 0.032 * 8000; % 32ms @ NB
leng_step = 0.010 * 8000; % 10ms @ NB
frameLen_process = 0.020 * 8000; % 20ms  @ NB
wind = hann(frameLen_process,'periodic'); % window length = processing frame length
K_fft = frameLen * 2; % NOT for prime Types

% - Load vectors
load('./data/type_3_cnn_output_ceps.mat');
load('./data/type_3_ceps_resi.mat'); 
load('./data/type_3_pha_ang.mat');

% - Waveform reconstruction frame-wise
for k = 1 : size(predictions,1), % num. of frames       
    
    % Form cepstral coefficients with also residuals
    ceps_coeff = [predictions(k,:).' ; ceps_resi(k,:).'];
    
    % Get time domain frame
    [s_rec_temp] = sub_func_recons_ceps_phase_timeSig_frame(ceps_coeff,ang_mat(k,:).',K_fft);
    s_rec(k,:) = s_rec_temp.';
    
    % 50% overlap-add
    indices_vor = 1 : frameLen_process/2;
    indices_nach = frameLen_process/2+1 : frameLen_process;
    if k == 2,
        x_fram_temp_nach = s_rec(1,indices_nach);
        x_fram_temp_vor = s_rec(2,indices_vor);

        s_rec_vec = x_fram_temp_nach + x_fram_temp_vor;
    elseif k>=3,
        x_fram_temp_nach = s_rec(k-1,indices_nach);
        x_fram_temp_vor = s_rec(k,indices_vor);

        s_rec_vec = [s_rec_vec, ...
            (x_fram_temp_nach + x_fram_temp_vor)];
    end
    
    % Display processing percentage!
    if mod(k,floor(size(predictions,1)/10)) == 0,
        disp([num2str(k) ' frames out of ' num2str(size(predictions,1)) ' frames finished!']);
        disp([num2str(k/size(predictions,1)*100) '% finished!']);
    end
end

% - Abandon/add last few samples if needed
legacy_dec_out = loadshort('./dataset/exapmle_s_g711_coded.raw');
if length(s_rec_vec) > length(legacy_dec_out),
    s_rec_vec = ...
        s_rec_vec(1:length(legacy_dec_out));
elseif length(s_rec_vec) < length(legacy_dec_out),
    s_rec_vec = ...
        [s_rec_vec, zeros(1,length(legacy_dec_out)-length(s_rec_vec))];
end

% - Convert to raw file and save
s_rec_vec = s_rec_vec * 2^15;
saveshort(s_rec_vec,'./dataset/cnn_postprocessed_8K_out.raw');

fs = 8000;
soundsc(s_rec_vec, fs)




