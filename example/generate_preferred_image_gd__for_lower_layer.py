# Demonstration codes for the usage of "prefer_img_gd"
# The codes will generate a preferred image for the target uints using gradient descent with momentum.

# import
import os
import pickle
import numpy as np
import PIL.Image
import caffe
import scipy.io as sio
from datetime import datetime

from cnnpref.utils import normalise_img, clip_extreme_pixel
from cnnpref.prefer_img_gd import generate_image

# average image of ImageNet
img_mean_fn = '../data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_fn)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# load cnn model
model_file = '../net/bvlc_alexnet/bvlc_alexnet.caffemodel'
prototxt_file = '../net/bvlc_alexnet/bvlc_alexnet.prototxt'
channel_swap = (2,1,0)
net = caffe.Classifier(prototxt_file, model_file, mean = img_mean, channel_swap = channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1,3,h,w)

# target layer
layer = 'conv1'

# target channels
num_of_ch = net.blobs[layer].data.shape[1]
num_of_img = 25
step = int(num_of_ch/num_of_img)
channel_list = range(0,num_of_ch,step)

# initial image for the optimization
initial_image = np.zeros((h,w,3),dtype='float32')

# make directory for saving the results
save_dir = '../result'
save_folder = __file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir,save_folder)
os.mkdir(save_path)

# options
opts = {
    
    'iter_n': 200, # the total number of iterations for gradient descend
    
    'disp_every': 1, # display the information on the terminal for every n iterations
    
    'save_intermediate': False, # save the intermediate or not
    'save_intermediate_every': 10, # save the intermediate for every n iterations
    'save_intermediate_path': save_path, # the path to save the intermediate
    
    'lr_start': 1., # learning rate
    'lr_end': 1.,
    
    'momentum_start': 0.001, # gradient with momentum
    'momentum_end': 0.001,
    
    'decay_start': 0.001, # pixel decay for each iteration
    'decay_end': 0.001,
    
    'image_blur': True, # Use image smoothing or not
    'sigma_start': 2.5, # the size of the gaussian filter for image smoothing
    'sigma_end': 0.5,
    
    'image_jitter': True, # use image jittering during
    'jitter_size': 4,
    
    'clip_small_norm': True,
    'clip_small_norm_every': 1,
    'n_pct_start': 5,
    'n_pct_end': 5,
    
    'clip_small_contribution': True,
    'clip_small_contribution_every': 1,
    'c_pct_start': 5,
    'c_pct_end':5,
    
    'initial_image': None, # the initial image for the optimization (setting to None will use random noise as initial image)
    }

# save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path,save_name),'w') as f:
    pickle.dump(opts,f)
    f.close()

# generate preferred image
for channel in channel_list:
    #
    print('')
    print('channel='+str(channel))
    print('')
    
    # target units
    feat_size = net.blobs[layer].data.shape[1:]
    y_index = int(feat_size[1]/2) # the unit in the center of feature map
    x_index = int(feat_size[2]/2) # the unit in the center of feature map
    feature_mask = np.zeros(feat_size)
    feature_mask[channel,y_index,x_index] = 1
    
    # weights for the target units
    feature_weight = np.zeros(feat_size, dtype=np.float32)
    feature_weight[:] = 1.
    
    #
    preferred_img = generate_image(net, layer, feature_mask, feature_weight=feature_weight, **opts)
    
    # save the results
    save_name = 'preferred_img' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.mat'
    sio.savemat(os.path.join(save_path,save_name),{'preferred_img':preferred_img})
    
    save_name = 'preferred_img' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.jpg'
    PIL.Image.fromarray(normalise_img(clip_extreme_pixel(preferred_img,pct=0.04))).save(os.path.join(save_path,save_name))

# end
print('Done!')
