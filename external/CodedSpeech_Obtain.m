%   Original (uncoded) speech loading, preprocessing, and en- & de-coding %
%   Input:  
%         1- Original (uncoded) speech example at 16KHz: exapmle_s_16k.raw% 
%         2- ITU-T G.191 compiled programs:  
%            filter.exe, sv56demo.exe, and g711demo.exe                   % 
%   Output:  
%         1- Coded speech  at 8 KHz: exapmle_s_g711_coded.raw             % 
%         2- Unoded speech at 8 KHz: exapmle_s_uncoded.raw                % 

clear; clc; 

% - Preprocessing
% FLAT filtering 
[~, ~]=system([' filter.exe ' ' -q FLAT1 ' ' ./dataset/exapmle_s_16k.raw ' ' ./dataset/exapmle_s_flat.raw ' ' 80 '], '-echo'); 
% Downsampling 
[~, ~]=system([' filter.exe ' ' -q -down HQ2 ' ' ./dataset/exapmle_s_flat.raw ' ' ./dataset/exapmle_s_flat_8k.raw ' ' 80 '],'-echo'); 
delete('./dataset/exapmle_s_flat.raw');
% Level Adjustment
[~, ~]=system([' sv56demo.exe ' ' -qq -lev -26 -sf 8000 ' ' ./dataset/exapmle_s_flat_8k.raw ' ' ./dataset/exapmle_s_uncoded.raw ' ' 80 '],'-echo'); 
delete('./dataset/exapmle_s_flat_8k.raw');
% - En- & de-coding
[~, ~]=system([' g711demo.exe ' ' A ' ' lilo ' ' ./dataset/exapmle_s_uncoded.raw ' ' ./dataset/test_bitstream_lo_g711.raw ' ],'-echo'); 
% delete('./dataset/exapmle_s_uncoded.raw');
[~, ~]=system([' g711demo.exe ' ' A ' ' loli ' ' ./dataset/test_bitstream_lo_g711.raw ' ' ./dataset/exapmle_s_g711_coded.raw ' ],'-echo'); 
delete('./dataset/test_bitstream_lo_g711.raw');




