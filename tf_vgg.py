import tensorflow as tf
import numpy as np

import pickle

with open('vgg16_state_dict.pickle', 'rb') as f:
    state_dict = pickle.load(f)


mean=np.array([0.485, 0.456, 0.406])
std=np.array([0.229, 0.224, 0.225])


def convert_weights(weight):
    # inverse to
    # torch: (out_channels, in_channels, kernel, kernel)
    # tensorflow: (kernel, kernel, in_channels, out_channels)

    # so looks like we need to
    # here tensorflow-> torch uses .permute(3, 2, 0, 1) # kh, kw, in, out -> out, in, kh, kw

    #https://github.com/huggingface/pytorch-pretrained-BigGAN/blob/master/pytorch_pretrained_biggan/convert_tf_to_pytorch.py
    if len(weight.shape) == 1: # biases
        weight = np.squeeze(weight)
    elif len(weight.shape) == 2: # Linear
        weight = np.transpose(weight)
    elif len(weight.shape) == 4:  # Convolutions
        weight = np.transpose(weight, [2, 3, 1, 0])
    return weight


def get_conv_filter(name):
    switcher = {
        'conv1_1':'features.0.weight',
        'conv1_2':'features.2.weight',
        'conv2_1':'features.5.weight',
        'conv2_2':'features.7.weight',
        'conv3_1':'features.10.weight',
        'conv3_2':'features.12.weight',
        'conv3_3':'features.14.weight',
        'conv4_1':'features.17.weight',
        'conv4_2':'features.19.weight',
        'conv4_3':'features.21.weight',
        'conv5_1':'features.24.weight',
        'conv5_2':'features.26.weight',
        'conv5_3':'features.28.weight'

        }
    f = state_dict[switcher[name]]

    f = convert_weights(f)

    #if name == 'conv1_2':
    #    print(np.mean(f))

    return f

def get_bias(name):
    switcher = {
        'conv1_1':'features.0.bias',
        'conv1_2':'features.2.bias',
        'conv2_1':'features.5.bias',
        'conv2_2':'features.7.bias',
        'conv3_1':'features.10.bias',
        'conv3_2':'features.12.bias',
        'conv3_3':'features.14.bias',
        'conv4_1':'features.17.bias',
        'conv4_2':'features.19.bias',
        'conv4_3':'features.21.bias',
        'conv5_1':'features.24.bias',
        'conv5_2':'features.26.bias',
        'conv5_3':'features.28.bias'

        }
    f = state_dict[switcher[name]]

    f = convert_weights(f)

    return f



def conv_layer(bottom, name):
   with tf.variable_scope(name):
       filt = get_conv_filter(name)

       conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME', name='conv2d')

       conv_biases = get_bias(name)
       bias = tf.nn.bias_add(conv, conv_biases, name='bias')
       return bias

def avg_pool(bottom, name):
    return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

# remember max pool indices to reuse them during backprop
def max_pool_with_argmax(net, ksize, strides):
  assert isinstance(ksize, list) or isinstance(ksize, int)
  assert isinstance(strides, list) or isinstance(strides, int)

  ksize = ksize if isinstance(ksize, list) else [1, ksize, ksize, 1]
  strides = strides if isinstance(strides, list) else [1, strides, strides, 1]

  with tf.name_scope('MaxPoolArgMax'):
    net, mask = tf.nn.max_pool_with_argmax(
      net,
      ksize=ksize,
      strides=strides,
      padding='SAME')
    return net, mask


def preprocess(rgb_255):
    rgb = rgb_255/255.0
    rgb = (rgb - mean)/std
    return rgb

def unprocess(rgb):
    rgb = (rgb*std + mean)*255.0
    return rgb



def vgg16(rgb, pool='max', use_masks=False, pool_masks={}, reuse=False):

    with tf.variable_scope('stylized_vgg') as scope:
        if reuse:
            scope.reuse_variables()

        masks = {}
        conv1_1 = conv_layer(rgb, "conv1_1")
        relu1_1 = tf.nn.relu(conv1_1)
        conv1_2 = conv_layer(relu1_1, "conv1_2")
        relu1_2 = tf.nn.relu(conv1_2)
        if pool=='max':
            if use_masks is True: # use stored pool mask
                mask = pool_masks['pool1']
                pool1 = tf.reshape(tf.gather(tf.reshape(relu1_2, [-1]), mask), mask.shape)
            else:
                pool1, mask = max_pool_with_argmax(relu1_2, 2, 2)
            masks['pool1'] = mask
        else:
            pool1 = avg_pool(relu1_2, 'pool1')

        conv2_1 = conv_layer(pool1, "conv2_1")
        relu2_1 = tf.nn.relu(conv2_1)
        conv2_2 = conv_layer(relu2_1, "conv2_2")
        relu2_2 = tf.nn.relu(conv2_2)
        if pool=='max':
            if use_masks is True: # use stored pool mask
                mask = pool_masks['pool2']
                pool2 = tf.reshape(tf.gather(tf.reshape(relu2_2, [-1]), mask), mask.shape)
            else:
                pool2, mask = max_pool_with_argmax(relu2_2, 2, 2)
            masks['pool2'] = mask
        else:
            pool2 = avg_pool(relu2_2, 'pool2')

        conv3_1 = conv_layer(pool2, "conv3_1")
        relu3_1 = tf.nn.relu(conv3_1)
        conv3_2 = conv_layer(relu3_1, "conv3_2")
        relu3_2 = tf.nn.relu(conv3_2)
        conv3_3 = conv_layer(relu3_2, "conv3_3")
        relu3_3 = tf.nn.relu(conv3_3)
        if pool=='max':
            if use_masks is True: # use stored pool mask
                mask = pool_masks['pool3']
                pool3 = tf.reshape(tf.gather(tf.reshape(relu3_3, [-1]), mask), mask.shape)
            else:
                pool3, mask = max_pool_with_argmax(relu3_3, 2, 2)
            masks['pool3'] = mask
        else:
            pool3 = avg_pool(relu3_3, 'pool3')

        conv4_1 = conv_layer(pool3, "conv4_1")
        relu4_1 = tf.nn.relu(conv4_1)
        conv4_2 = conv_layer(relu4_1, "conv4_2")
        relu4_2 = tf.nn.relu(conv4_2)
        conv4_3 = conv_layer(relu4_2, "conv4_3")
        relu4_3 = tf.nn.relu(conv4_3)
        if pool=='max':
            if use_masks is True: # use stored pool mask
                mask = pool_masks['pool4']
                pool4 = tf.reshape(tf.gather(tf.reshape(relu4_3, [-1]), mask), mask.shape)
            else:
                pool4, mask = max_pool_with_argmax(relu4_3, 2, 2)
            masks['pool4'] = mask
        else:
            pool4 = avg_pool(relu4_3, 'pool4')

        conv5_1 = conv_layer(pool4, "conv5_1")
        relu5_1 = tf.nn.relu(conv5_1)
        conv5_2 = conv_layer(relu5_1, "conv5_2")
        relu5_2 = tf.nn.relu(conv5_2)
        conv5_3 = conv_layer(relu5_2, "conv5_3")
        relu5_3 = tf.nn.relu(conv5_3)

        return {
            'conv1_1':conv1_1,
            'conv1_2':conv1_2,
            'conv2_1':conv2_1,
            'conv2_2':conv2_2,
            'conv3_1':conv3_1,
            'conv3_2':conv3_2,
            'conv3_3':conv3_3,
            'conv4_1':conv4_1,
            'conv4_2':conv4_2,
            'conv4_3':conv4_3,
            'conv5_1':conv5_1,
            'conv5_2':conv5_2,
            'conv5_3':conv5_3
            }, masks


print('VGG16 loaded')
