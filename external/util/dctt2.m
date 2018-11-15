function c = dctt2(x, c_dim)
% 
% DCT2 - Discrete Cosine Transform Type II 
%
% Usage: c = dct2(x, c_dim);
% 
% Inputs:
%        x              : filter bank feature vectors (x_dim  x  num_x)
%        c_dim          : desired dimension of DCT output vectors c
%                         0 < c_dim <= x_dim
% Outputs:
%        c              : DCT coefficients (c_dim  x  num_x)
%
% This function computes the type II DCT using (5.93) in Acero's book
% "Spoken Language Processing". See p. 228 there for the trick using  the
% (I)DFT.
%
% Technische Universität Braunschweig, IfN, 2008 - 08 - 14 (version 1.0)
% (c) Prof. Dr.-Ing. Tim Fingscheidt
%--------------------------------------------------------------------------

%--- inits
x_dim = size(x,1);
num_x = size(x,2);

c = zeros(c_dim,num_x);

scalefactor = x_dim * exp(1i*pi*(0:1:2*x_dim-1)*0.5/x_dim);
scalefactor = scalefactor(:);

for fv=1:num_x,
    temp    = scalefactor .* ifft([x(:,fv);flipud(x(:,fv))]);
    c(:,fv) = real(temp(1:c_dim));      % Im{} is anyway approx. = 0
end;

