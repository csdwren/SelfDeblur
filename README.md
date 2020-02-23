## Neural Blind Deconvolution Using Deep Priors 
[[arxiv](https://arxiv.org/abs/1908.02197)] [[supp](https://csdwren.github.io/papers/SelfDeblur_supp.pdf)]


### Introduction
Blind deconvolution is a classical yet challenging low-level vision problem with many real-world applications.
Traditional maximum a posterior (MAP) based methods rely heavily on fixed and handcrafted priors that certainly are insufficient in characterizing clean images and blur kernels, and usually adopt specially designed alternating minimization to avoid trivial solution.
In contrast, existing deep motion deblurring networks learn from massive training images the mapping to clean image or blur kernel, but are limited in handling various complex and large size blur kernels.
Motivated by deep image prior (DIP) [1], we in this paper present two generative networks for respectively modeling the deep priors of clean image and blur kernel, and propose an unconstrained neural optimization solution to blind deconvolution (SelfDeblur).
Experimental results show that our SelfDeblur can achieve notable quantitative gains as well as more visually plausible deblurring results in comparison to state-of-the-art blind deconvolution methods on benchmark datasets and real-world blurry images.


## Prerequisites
- Python 3.6, PyTorch >= 0.4 
- Requirements: opencv-python, tqdm
- Platforms: Ubuntu 16.04, TITAN V, cuda-10.0 & cuDNN v-7.5
- MATLAB for computing [evaluation metrics](statistic/)


## Datasets

SelfDeblur is evaluated on datasets of Levin et al. [2] and Lai et al. [3]. 
Please download the testing datasets from [BaiduYun](https://pan.baidu.com/s/1FRqEzhkfs0ZIy0TuZm7Cnw)
or [OneDrive](https://1drv.ms/u/s!An-BNLJWOClldZsSDPju_HHf2d4?e=9LEHmi), 
and place the unzipped folders into `./datasets/`.


## Getting Started

### 1. Run SelfDeblur


(1) SelfDeblur on Levin dataset. The code has been improved, and usually can achieve better retults than those reported in the paper.
```bash
python selfdeblur_levin.py 
```

(2) SelfDeblur on Lai dataset, where blurry images have firstly been converted to their Y channel. Several images may converge to "black" image, but their blur kernels are good. I will check why this happened. In these cases, you need to run `selfdeblur_nonblind.py` to generate good deblurring results.
```bash
python selfdeblur_lai.py 
python selfdeblur_nonblind.py --data_path path_to_your_results_with_pretreind_Gk # Optional nonblind SelfDeblur. Given pretrained Gk, only update Gx while fixing Gk.
```

(3) Handle color images in YCbCr space. 2500 iterations are adopted. If you need better texture details, more iterations will help. 
```bash
python selfdeblur_ycbcr.py # Deblur several color images in `./datasets/real/`.
```


_*In current SelfDeblur code, TV regularization has been removed. The improved code is more robust to blur kernel estimation. But for some images with high level noises and non-uniform blurry images, the deblurring results may suffer from ringing effects due to our uniform convolution-based loss function. In this case, adding TV regularization to SelfDeblur loss function or running another nonblind deblur method may be an choice._

All the deblurring results and deep models are also available. Please read [results/levin/readme.docx](/results/levin/readme.docx) and [results/lai/readme.docx](results/lai/readme.docx) for the details. 
You can place the downloaded results into `./results/`, and directly compute all the [evaluation metrics](statistic/) in this paper.  

### 2. Evaluation metrics

We also provide the MATLAB scripts to compute the average PSNR and SSIM values reported in the paper.
 

```Matlab
 cd ./statistic
 run statistic_levin.m 
 run statistic_lai.m 
```


SelfDeblur succeeds in simultaneously estimating blur kernel and generating clean image with finer texture details. 
<img src="results/demo/levin.png" width="800px"/>
<img src="results/demo/lai.jpg" width="800px"/> 


## References
[1] D. Ulyanov, A. Vedaldi, and V. Lempitsky. Deep image prior. In IEEE CVPR 2018. 

[2] A. Levin, Y. Weiss, F. Durand, and W. T. Freeman. Understanding and evaluating blind deconvolution algorithms. In IEEE CVPR 2009. 

[3] W.-S. Lai, J.-B. Huang, Z. Hu, N. Ahuja, and M.-H. Yang. A comparative study for single image blind deblurring. In IEEE CVPR 2016.




