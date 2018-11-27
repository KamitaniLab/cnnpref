# Demonstration codes for the usage of "prefer_img_lbfgs"
# The codes will generate preferred image for the target uints uints using L-BFGS-B.

# import
import os
import pickle
import numpy as np
import PIL.Image
import caffe
import scipy.io as sio
from datetime import datetime

from cnnpref.utils import normalise_img, clip_extreme_pixel
from cnnpref.prefer_img_lbfgs import generate_image

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
layer = 'fc8'

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
    
    'maxiter': 500, # the maximum number of iterations
    
    'disp': True, # display or not the information on the terminal
    
    'save_intermediate': False, # save the intermediate or not
    'save_intermediate_every': 50, # save the intermediate for every n iterations
    'save_intermediate_path': save_path, # the path to save the intermediate
    
    'initial_image': initial_image, # the initial image for the optimization (setting to None will use random noise as initial image)
    
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
    feature_mask = np.zeros(feat_size)
    feature_mask[channel] = 1
    
    # weights for the target units
    feature_weight = np.zeros(feat_size, dtype=np.float32)
    feature_weight[:] = 10.
    
    #
    preferred_img = generate_image(net, layer, feature_mask, feature_weight=feature_weight, **opts)
    
    # save the results
    save_name = 'preferred_img' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.mat'
    sio.savemat(os.path.join(save_path,save_name),{'preferred_img':preferred_img})
    
    save_name = 'preferred_img' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.jpg'
    PIL.Image.fromarray(normalise_img(clip_extreme_pixel(preferred_img,pct=0.04))).save(os.path.join(save_path,save_name))

# end
print('Done!')
