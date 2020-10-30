import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

class CONFIG:
    IMAGE_WIDTH = 400
    IMAGE_HEIGHT = 300
    COLOR_CHANNELS = 3
    MEANS = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
    VGG_MODEL = 'pretrained-model/imagenet-vgg-verydeep-19.mat'
    STYLE_IMAGE = 'images/design_1.jpg'
    CONTENT_IMAGE = 'images/shirt_0.jpg'
    OUTPUT_DIR = 'output/'

def weights(layer, expected_layer_name):
    wb = vgg_layers[0][layer][0][0][2]
    W = wb[0][0]
    b = wb[0][1]
    layer_name = vgg_layers[0][layer][0][0][0][0]
    assert layer_name == expected_layer_name
    return W, b

def conv2d(prev_layer, layer, layer_name):
    W, b = weights(layer, layer_name)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

def conv2d_relu(prev_layer, layer, layer_name):
    return tf.nn.relu((conv2d(prev_layer, layer, layer_name)))

def load_vgg_model(path):
    vgg = scipy.io.loadmat(path)
    vgg_layers = vgg['layers']
    
    graph = {}
    graph['input']   = tf.Variable(np.zeros((1, CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH, CONFIG.COLOR_CHANNELS)), dtype = 'float32')
    graph['conv1_1']  = conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2']  = conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = tf.nn.avg_pool(graph['conv1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv2_1']  = conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2']  = conv2d_relu(graph['conv2_2'], 7, 'conv2_2')
    graph['avgpool2'] = tf.nn.avg_pool(graph['conv1_2'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv3_1']  = conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2']  = conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3']  = conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4']  = conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = tf.nn.avg_pool(graph['conv3_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv4_1']  = conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2']  = conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3']  = conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4']  = conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = tf.nn.avg_pool(graph['conv4_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    graph['conv5_1']  = conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2']  = conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3']  = conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4']  = conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = tf.nn.avg_pool(graph['conv5_4'], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return graph

def reshape_and_normalize_image(image):
    image = np.reshape(image, ((1,) + image.shape))
    return image - CONFIG.MEANS

def save_image(path, image):
    image = image + CONFIG.MEANS
    scipy.misc.imsave(path, np.clip(image[0], 0, 255).astype('uint8'))
