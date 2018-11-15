function [ceps_coeff_alter,ceps_env,ceps_resi,ang_s,s_frame_w,ind_outlier_frame] = sub_func_time_fram2ceps_coeff_pha(s_frame,wind,K_fft)
%%% Output: Colum vector

% --- 32(num_ceps_coef)-256(frame length)-512(K points FFT)-8KHz, 64-512-16KHz, 128-1024-32KHz; 
num_ceps_coef = K_fft/2 * 0.125; 
%--- Windowing
s_frame_w = s_frame .* wind;


%--- FFT per frame
spec_s = fft(s_frame_w,K_fft);
%--- Angles of the fft coefficients
ang_s = angle(spec_s);

%--- DCT-II type cepstral coefficients
K_ceps = K_fft;
fft_ind_vec = [0 : K_fft-1].';
ind_ceps_vec = [1 : K_ceps].';
abs_spec_s = abs(spec_s);
% -- Check if outlier(!) exit? -- %
check_outlier = sum(find(abs_spec_s == 0));
ind_outlier_frame = 0;
if check_outlier ~= 0,
    [ind_a,ind_b] = find(~abs_spec_s);
    abs_spec_s(ind_a,ind_b) = eps; 
    
    % document this frame
    ind_outlier_frame = 1;

end
% --      End Check      -- %
a_part_s = log10(abs_spec_s); 
ceps_coeff_alter = dctt2(a_part_s, K_fft);

%--- Select part of all cepstral coefficients 
ceps_env = ceps_coeff_alter(1:num_ceps_coef);
ceps_resi = ceps_coeff_alter(num_ceps_coef+1:end);
