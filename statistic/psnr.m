function [ PSNR ] = psnr( f1,f2 )
%PSNR Summary of this function goes here
%   Detailed explanation goes here


% % MSE = E( (img£­Eimg)^2 ) 
% %     = SUM((img-Eimg)^2)/(M*N);
% function ERMS = erms(f1, f2)
% %
% e = double(f1) - double(f2);
% [m, n] = size(e);
% ERMS = sqrt(sum(e.^2)/(m*n));
% % PSNR=10log10(M*N/MSE); 
% function PSNR = psnr(f1, f2)
%
k=1;
if max(f1(:))>2
k = 8;
end

%k
fmax = 2.^k - 1;
a = fmax.^2;
e = double(f1) - double(f2);
[m, n] = size(e);
MSE=sum(sum(e.^2))/(m*n);
PSNR = 10*log10(a/MSE);
end

