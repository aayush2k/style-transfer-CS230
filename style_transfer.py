import os
import sys
import scipy.io
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import tensorflow as tf
%matplotlib inline

def content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled =  tf.reshape(a_C, shape=[m, n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[m, n_H * n_W, n_C])
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    return J_content

def gram_matrix(A):
    GA = tf.matmul(A, tf.transpose(A))    
    return GA

def layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    GS = gram_matrix(tf.reshape(tf.transpose(a_S), shape=[n_C, n_H * n_W]))
    GG = gram_matrix(tf.reshape(tf.transpose(a_G), shape=[n_C, n_H * n_W]))
    J_style_layer = (1 / (4 * (n_H*n_W)*(n_H*n_W) * n_C*n_C)) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    return J_style_layer

def style_cost(model, STYLE_LAYERS):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha*J_content + beta*J_style    
    return J

def model_nn(sess, input_image, num_iterations = 200):
    sess.run(tf.global_variables_initializer())
    sess.run(model["input"].assign(input_image))
    
    for i in range(num_iterations):
        sess.run(train_step)
        generated_image = sess.run(model["input"])
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("Total cost = " + str(Jt))
            print("Content cost = " + str(Jc))
            print("Style cost = " + str(Js))
            save_image("output/" + str(i) + ".png", generated_image)
   
    save_image('output/generated_image.jpg', generated_image)
    return generated_image

def main():
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    
    STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    content_image = reshape_and_normalize_image(scipy.misc.imread("images/shirt_0.jpg"))
    style_image = reshape_and_normalize_image(scipy.misc.imread("images/design_1.jpg"))
    model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
    sess.run(model['input'].assign(content_image))
    out = model['conv4_2']
    a_C = sess.run(out)
    a_G = out
    J_content = content_cost(a_C, a_G)
    sess.run(model['input'].assign(style_image))
    J_style = style_cost(model, STYLE_LAYERS)
    J = total_cost(J_content, J_style, alpha = 10, beta = 40)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)
    model_nn(sess, generated_image)

if __name__ == "__main__":
    main()
