function [psnrs,ssims,tI1]=comp_upto_shift(I1,I2,maxshift)
%function [ssde,tI1]=comp_upto_shift(I1,I2)
%  compute sum of square differences between two images, after
%  finding the best shift between them. need to account for shift
%  because the kernel reconstruction is shift invariant- a small 
%  shift of the image and kernel will not effect the likelihood score.
%Input:
%I1,I2-images to compare
%Output:
%ssde-sum of square differences
%tI1-image I1 at best shift toward I2
%
%Writen by: Anat Levin, anat.levin@weizmann.ac.il (c)

[N1,N2]=size(I1);


%maxshift=5;
shifts=[-maxshift:0.25:maxshift];




I2=I2(16:end-15,16:end-15);
I1=I1(16-maxshift:end-15+maxshift,16-maxshift:end-15+maxshift);
[N1,N2]=size(I2);
[gx,gy]=meshgrid([1-maxshift:N2+maxshift],[1-maxshift:N1+maxshift]);

[gx0,gy0]=meshgrid([1:N2],[1:N1]);
 


for i=1:length(shifts)
   for j=1:length(shifts)

     gxn=gx0+shifts(i);
     gyn=gy0+shifts(j);
     tI1=interp2(gx,gy,I1,gxn,gyn);
   
     ssdem(i,j)=sum(sum((tI1-I2).^2));
    
   end
end

ssde=min(ssdem(:));
[i,j]=find(ssdem==ssde);

gxn=gx0+shifts(i);
gyn=gy0+shifts(j);
tI1=interp2(gx,gy,I1,gxn,gyn);
psnrs = psnr(255*tI1,255*I2);
ssims = ssim(255*tI1,255*I2);