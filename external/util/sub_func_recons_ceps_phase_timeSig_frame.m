function [s_fram_under_ola] = sub_func_recons_ceps_phase_timeSig_frame(ceps_coeff_whole,ang_s,K_fft)

% Input: ceps_coeff_whole - colum vector
%--- IDCT -> power spectrum 
% s_idct = idctt2(ceps_coeff_whole, K_fft); % s_idct should be symmetric

%%%% Use matrix multiple instead           %%%
%%%% c -> ceps_coeff_whole, x_dim -> K_fft, x -> s_idct %%%%
K = K_fft;
M = size(ceps_coeff_whole,1);
m_vec = [1:M-1].';
k_prim_vec = [0:K-1] + 0.5;
A_vec = ceps_coeff_whole(2:M).';
B_mat = cos(pi/K*m_vec*k_prim_vec);
x_prim_vec = A_vec * B_mat;
x_vec = (2 * x_prim_vec + ceps_coeff_whole(1) ) / K;
s_idct = x_vec.';
%%%% End insteading %%% 

s_idct_exp = 10.^s_idct; % 10.^s_idct;   exp(1).^s_idct;
%--- Use s_idct and phase to reconstruct FFT coefficients
spec_s_rec = s_idct_exp.*exp(1i*ang_s);  % s power spectrum, s phase
%--- Re[IFFT(.)] from FFT coefficients
s_rec_temp = real(ifft(spec_s_rec,K_fft));
%--- Half K_fft length in time-domain
s_fram_under_ola = s_rec_temp(1:K_fft/2); % because FFT point = 2*frame length