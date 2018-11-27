# Demonstration codes for the usage of "prefer_img_dgn_gd"
# The codes will generate preferred image for the target uints using gradient descent with momentum with constraint via a deep generator net.

# import
import os
import pickle
import numpy as np
import PIL.Image
import caffe
import scipy.io as sio
from datetime import datetime

from cnnpref.utils import normalise_img, clip_extreme_pixel, create_receptive_field_mask
from cnnpref.prefer_img_dgn_gd import generate_image

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

# load the generator net
model_file = '../net/generator_for_inverting_fc6/generator.caffemodel'
prototxt_file = '../net/generator_for_inverting_fc6/generator.prototxt'
net_gen = caffe.Net(prototxt_file,model_file,caffe.TEST)
input_layer_gen = 'feat' # input layer for generator net
output_layer_gen = 'generated' # output layer for generator net

# feature size for input layer of the generator net
feat_size_gen = net_gen.blobs[input_layer_gen].data.shape[1:]
num_of_unit = np.prod(feat_size_gen)

# upper bound for input layer of the generator net
bound_file = '../data/act_range/3x/fc6.txt'
upper_bound = np.loadtxt(bound_file,delimiter=' ',usecols=np.arange(0,num_of_unit),unpack=True)
upper_bound = upper_bound.reshape(feat_size_gen)

# make folder for saving the results
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
    
    'lr_start': 0.1, # learning rate
    'lr_end': 0.01,
    
    'momentum_start': 0.1, # gradient with momentum
    'momentum_end': 0.1,
    
    'decay_start': 0.001, # decay for the features of the input layer of the generator after each iteration
    'decay_end': 0.001,
    
    'input_layer_gen': input_layer_gen, # name of the input layer of the generator (str)
    'output_layer_gen': output_layer_gen, # name of the output layer of the generator (str)
    
    'feat_upper_bound': upper_bound, # set the upper boundary for the input layer of the generator
    'feat_lower_bound': 0., # set the lower boundary for the input layer of the generator
    
    'initial_gen_feat': None, # the initial features of the input layer of the generator (setting to None will use random noise as initial features)
    
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
    
    # create image mask for the receptive fields of the target units
    img_mask = create_receptive_field_mask(net, layer, feature_mask)
    
    #
    preferred_img = generate_image(net_gen, net, layer, feature_mask, feature_weight=feature_weight, **opts)
    
    # save the results
    save_name = 'preferred_img' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.mat'
    sio.savemat(os.path.join(save_path,save_name),{'preferred_img':preferred_img})
    
    save_name = 'preferred_img' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.jpg'
    PIL.Image.fromarray(normalise_img(clip_extreme_pixel(preferred_img,pct=0.04))).save(os.path.join(save_path,save_name))
    
    save_name = 'preferred_img_masked' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.mat'
    sio.savemat(os.path.join(save_path,save_name),{'preferred_img':preferred_img * img_mask})
    
    save_name = 'preferred_img_masked' + '_layer_' + str(layer) + '_channel_' + str(channel) + '.jpg'
    PIL.Image.fromarray(normalise_img(clip_extreme_pixel(preferred_img,pct=0.04)) * img_mask).save(os.path.join(save_path,save_name))

# end
print('Done!')

print(net.params[layer][0].data.shape)

