
clear all

otherspath='../results/levin/others/';
gt_path='../results/levin/groundtruth/';
groundk=fullfile(otherspath,'groundk/');
cholee=fullfile(otherspath,'ChoAndLee/');
xujia=fullfile(otherspath,'xuAndJia/');
krishnan=fullfile(otherspath,'Krishnan/');
levin=fullfile(otherspath,'Levin/');
sun=fullfile(otherspath,'Sun/');
dark = fullfile(otherspath,'dark/');
zuo = fullfile(otherspath,'zuo/');

struct_model = {
     struct('name','groundk','path',groundk),...
     struct('name','dark','path',dark),...
     struct('name','levin','path',levin),...
     struct('name','xujia','path',xujia),...
     struct('name','cholee','path',cholee),...
     struct('name','krishnan','path',krishnan),...
     struct('name','sun','path',sun),...
     struct('name','zuo','path',zuo),...
    };
nmodel = length(struct_model);

nimgs = 4;
nkernels=8;
maxshift=10; %Usually maxshift=5 is enough. If you find very low PSNR and SSIM for images with visually good results, maxshift should be set as a larger value. 


for nnn = 1:nmodel
    modelpath = struct_model{nnn}.path;
    for iii=1:nimgs
        for jjj=1:nkernels
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            imgpath = fullfile(gt_path,sprintf('im%d.png',iii));
            x_true=im2double(imread(imgpath));%x_true
            
            %our
            imgpath = fullfile(modelpath,sprintf('im%d_kernel%d_img.png',iii,jjj));
            deblur=im2double(imread(imgpath));%deblur
            
            [tp,ts,tI] = comp_upto_shift(deblur,x_true); %tI is the aligned and cropped image. 
            
            imgpath = fullfile(groundk,sprintf('im%d_kernel%d_img.png',iii,jjj));
            deblur=im2double(imread(imgpath));%deblur
            [~,~,tI_groundk] = comp_upto_shift(deblur,x_true,maxshift);
            
            te = norm(tI-x_true(16:end-15,16:end-15))/norm(tI_groundk-x_true(16:end-15,16:end-15));
            
            imgpath = fullfile(modelpath,'aligned');
            if ~exist(imgpath,'dir')
                mkdir(imgpath);
            end
            imwrite(tI, fullfile(imgpath,sprintf('im%d_kernel%d_img.png',iii,jjj)))
            
            psnrs(iii,jjj)=tp;ssims(iii,jjj)=ts;errorRatios(iii,jjj)=te;
            
            fprintf('img=%d kernel=%d: psnr=%6.4f, ssim=%6.4f, errorRatio=%6.4f\n',iii,jjj,tp,ts,te);
            
        end
        
    end
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f, errorRatio=%6.4f\n',struct_model{nnn}.name,mean(psnrs(:)),mean(ssims(:)),mean(errorRatios(:)));
    
    save(fullfile(struct_model{nnn}.path,'psnrs.mat'),'psnrs');
    save(fullfile(struct_model{nnn}.path,'ssims.mat'),'ssims');
    save(fullfile(struct_model{nnn}.path,'errorRatios.mat'),'errorRatios');
    
    
end

%%
selfdeblur='../results/levin/SelfDeblur/imgs/';
struct_model = {
    struct('name','selfdeblur','path',selfdeblur),...
    };
nmodel = length(struct_model);

nimgs = 4;
nkernels=8;

for nnn = 1:nmodel
    modelpath = struct_model{nnn}.path;
    for iii=1:nimgs
        for jjj=1:nkernels
            %         fprintf('img=%d,kernel=%d\n',iii,jjj);
            imgpath = fullfile(gt_path,sprintf('im%d.png',iii));
            x_true=im2double(imread(imgpath));%x_true
            
            %our
            imgpath = fullfile(modelpath,sprintf('im%d_kernel%d_img_x.png',iii,jjj));
            deblur=im2double(imread(imgpath));%deblur
            
            [tp,ts,tI] = comp_upto_shift(deblur,x_true);
            
            imgpath = fullfile(groundk,sprintf('im%d_kernel%d_img.png',iii,jjj));
            deblur=im2double(imread(imgpath));%deblur
            [~,~,tI_groundk] = comp_upto_shift(deblur,x_true);
            
            te = norm(tI-x_true(16:end-15,16:end-15))/norm(tI_groundk-x_true(16:end-15,16:end-15));
            
            imgpath = fullfile(modelpath,'aligned');
            if ~exist(imgpath,'dir')
                mkdir(imgpath);
            end
            imwrite(tI, fullfile(imgpath,sprintf('im%d_kernel%d_img.png',iii,jjj)))
            
            psnrs(iii,jjj)=tp;ssims(iii,jjj)=ts;errRatios(iii,jjj)=te;
            
            fprintf('img=%d kernel=%d: psnr=%6.4f, ssim=%6.4f, errorRatio=%6.4f\n',iii,jjj,tp,ts,te);
            
        end
        
    end
    
    save(fullfile(modelpath,'psnrs.mat'),'psnrs');
    save(fullfile(modelpath,'ssims.mat'),'ssims');
    save(fullfile(modelpath,'errRatios.mat'),'errRatios');
    
    fprintf('%s: psnr=%6.4f, ssim=%6.4f, errRatio=%6.4f\n',struct_model{nnn}.name,mean(psnrs(:)),mean(ssims(:)),mean(errRatios(:)));
    
end




