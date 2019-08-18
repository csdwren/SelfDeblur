from .common_utils import *

def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])
    
    bbox = [
            int((target_size[0] - img_np.shape[1]) / 2),
            int((target_size[1] - img_np.shape[2]) / 2),
            int((target_size[0] + img_np.shape[1]) / 2),
            int((target_size[1] + img_np.shape[2]) / 2),
    ]
    
    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np
    
    return img_out





def load_img(fname):

    img_orig_pil, img_orig_np = get_image(fname, -1)


    img_HR_np = pil_to_np(img_HR_pil)

    print('blurry image resolutions: %s' % (str(img_HR_pil.size)))

    return img_HR_np



def get_baselines(img_LR_pil, img_HR_pil):
    '''Gets `bicubic`, sharpened bicubic and `nearest` baselines.'''
    img_bicubic_pil = img_LR_pil.resize(img_HR_pil.size, Image.BICUBIC)
    img_bicubic_np = pil_to_np(img_bicubic_pil)

    img_nearest_pil = img_LR_pil.resize(img_HR_pil.size, Image.NEAREST)
    img_nearest_np = pil_to_np(img_nearest_pil)

    img_bic_sharp_pil = img_bicubic_pil.filter(PIL.ImageFilter.UnsharpMask())
    img_bic_sharp_np = pil_to_np(img_bic_sharp_pil)

    return img_bicubic_np, img_bic_sharp_np, img_nearest_np



def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))
