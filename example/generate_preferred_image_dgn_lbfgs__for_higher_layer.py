# Demonstration codes for the usage of "prefer_img_dgn_lbfgs"
# The codes will generate preferred image for the target uints using L-BFGS-B with constraint via a deep generator net.

# import
import os
import pickle
import numpy as np
import PIL.Image
import caffe
import scipy.io as sio
from datetime import datetime

from cnnpref.utils import normalise_img, clip_extreme_pixel, create_receptive_field_mask
from cnnpref.prefer_img_dgn_lbfgs import generate_image

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

# gen_feat_bounds
gen_feat_bounds = []
for j in xrange(num_of_unit):
    gen_feat_bounds.append((0.,upper_bound[j])) # the lower bound is 0
# gen_feat_bounds = []
# for j0 in xrange(gen_feat_size[0]):
    # for j1 in xrange(gen_feat_size[1]):
        # for j2 in xrange(gen_feat_size[2]):
            # gen_feat_bounds.append((0.,upper_bound[j0]))

# make folder for saving the results
save_dir = '../result'
save_folder = __file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir,save_folder)
os.mkdir(save_path)

# reconstruction options
opts = {
    
    'maxiter':500, # the maximum number of iterations
    
    'disp':True, # print or not the information on the terminal
    
    'save_intermediate': True, # save the intermediate or not
    'save_intermediate_every': 10, # save the intermediate for every n iterations
    'save_intermediate_path': save_path, # the path to save the intermediate
    
    'input_layer_gen': input_layer_gen, # name of the input layer of the generator (str)
    'output_layer_gen': output_layer_gen, # name of the output layer of the generator (str)
    
    'gen_feat_bounds':gen_feat_bounds, # set the boundary for the input layer of the generator
    
    'initial_gen_feat':None, # the initial features of the input layer of the generator (setting to None will use random noise as initial features)
    
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

