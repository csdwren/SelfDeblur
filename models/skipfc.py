import torch
import torch.nn as nn
from .common import *

def skipfc(num_input_channels=2, num_output_channels=3, 
           num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
           filter_size_down=3, filter_size_up=1, filter_skip_size=1,
           need_sigmoid=True, need_bias=True, 
           pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
           need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
#    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
#
#    n_scales = len(num_channels_down)
#
#    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
#        upsample_mode   = [upsample_mode]*n_scales
#
#    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
#        downsample_mode   = [downsample_mode]*n_scales
#    
#    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
#        filter_size_down   = [filter_size_down]*n_scales
#
#    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
#        filter_size_up   = [filter_size_up]*n_scales
#
#    last_scale = n_scales - 1 
#
#    cur_depth = None
#
#    model = nn.Sequential()
#    model_tmp = model
#
#    input_depth = num_input_channels
#    for i in range(len(num_channels_down)):
#
#        deeper = nn.Sequential()
#        skip = nn.Sequential()
#
#        if num_channels_skip[i] != 0:
#            model_tmp.add(Concat(1, skip, deeper))
#        else:
#            model_tmp.add(deeper)
#        
#        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))
#
#        if num_channels_skip[i] != 0:
#            skip.add(nn.Linear(input_depth, num_channels_skip[i]))
#            skip.add(bn(num_channels_skip[i]))
#            skip.add(act(act_fun))
#            
#        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))
#
#        deeper.add(nn.Linear(input_depth, num_channels_down[i]))
#        deeper.add(bn(num_channels_down[i]))
#        deeper.add(act(act_fun))
#
#        deeper.add(nn.Linear(num_channels_down[i], num_channels_down[i]))
#        deeper.add(bn(num_channels_down[i]))
#        deeper.add(act(act_fun))
#
#        deeper_main = nn.Sequential()
#
#        if i == len(num_channels_down) - 1:
#            # The deepest
#            k = num_channels_down[i]
#        else:
#            deeper.add(deeper_main)
#            k = num_channels_up[i + 1]
#
#        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))
#
#        model_tmp.add(nn.Linear(num_channels_skip[i] + k, num_channels_up[i]))
#        model_tmp.add(bn(num_channels_up[i]))
#        model_tmp.add(act(act_fun))
#
#
#        if need1x1_up:
#            model_tmp.add(nn.Linear(num_channels_up[i], num_channels_up[i]))
#            model_tmp.add(bn(num_channels_up[i]))
#            model_tmp.add(act(act_fun))
#
#        input_depth = num_channels_down[i]
#        model_tmp = deeper_main
#
#    model.add(nn.Linear(num_channels_up[0], num_output_channels))
#    if need_sigmoid:
#        model.add(nn.Softmax())
#
#    return model

    model = nn.Sequential()
    model.add(nn.Linear(num_input_channels, num_channels_down[0],bias=True))
    model.add(nn.ReLU6())
#    model.add(nn.Tanh())
#    model.add(nn.Linear(num_channels_down[0], num_channels_down[0],bias=True))
#    model.add(nn.ReLU6())
#    model.add(nn.Linear(num_channels_down[0], num_channels_down[0],bias=True))
#    model.add(nn.ReLU6())
#    model.add(nn.Linear(num_channels_down[0], num_channels_down[0],bias=True))
#    model.add(nn.ReLU6())
#    model.add(nn.Linear(num_channels_down[0], num_channels_down[0],bias=True))
#    model.add(nn.ReLU6())
#    model.add(nn.Linear(num_channels_down[0], num_channels_down[0],bias=True))
#    model.add(nn.ReLU6())
#    model.add(nn.Linear(num_channels_down[0], num_channels_down[1]))
#    model.add(act(act_fun))
#    model.add(nn.Linear(num_channels_down[1], num_channels_down[2]))
#    model.add(act(act_fun))
#    model.add(nn.Linear(num_channels_down[2], num_channels_down[3]))
#    model.add(act(act_fun))
#    model.add(nn.Linear(num_channels_down[3], num_channels_up[3]))
#    model.add(act(act_fun))
#    model.add(nn.Linear(num_channels_up[3], num_channels_up[2]))
#    model.add(act(act_fun))
#    model.add(nn.Linear(num_channels_up[2], num_channels_up[1]))
#    model.add(act(act_fun))
#    model.add(nn.Linear(num_channels_up[1], num_channels_up[0]))
#    model.add(act(act_fun))
    model.add(nn.Linear(num_channels_down[0], num_output_channels))
#    model.add(nn.ReLU())
    model.add(nn.Softmax())
#    model.add(nn.Threshold(0.00001, 0))
    return model











