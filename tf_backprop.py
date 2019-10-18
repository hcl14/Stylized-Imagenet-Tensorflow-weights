import tensorflow as tf
import numpy as np
from PIL import Image
from tqdm import tqdm

from tf_vgg import vgg16 as stylized_vgg16, preprocess, unprocess

target_image = np.array(Image.open('003-5-gthumb-gwdata1200-ghdata1200-gfitdatamax.jpg_0.png').convert('RGB'), dtype=np.float32)[np.newaxis,...]
target_image = preprocess(target_image)


patch_size = target_image.shape[1]

# it will be just ones
data =  tf.placeholder(dtype=tf.float32, shape=[1, patch_size, patch_size, 3], name='data_in')

# target image
data_src =  tf.placeholder(dtype=tf.float32, shape=[1, patch_size, patch_size, 3], name='data_src')




### define input data

arr = np.zeros([1, patch_size, patch_size, 3], dtype=np.float32)

def reconstructed_image(input, reuse=False):
    with tf.variable_scope('reconstructed', reuse=reuse):
        coefs = tf.get_variable("reconstructed_coefs", [1, patch_size, patch_size, 3], initializer=tf.constant_initializer(arr)) #tf.random_normal_initializer())
        output = input * coefs
        tf.identity(output, name="output_image")

    return output


img_rec = reconstructed_image(data)

img_input = img_rec
img_src = data_src


# they work well together: remembered max pooling and avg pooling. AVG pushes all the values, MAX highlights the right one

vgg_src, masks = stylized_vgg16(img_src, pool='max')
vgg_net, _ = stylized_vgg16(img_input, pool='max',  use_masks=True, pool_masks=masks, reuse=True)


vgg_src1,_ = stylized_vgg16(img_src, pool='avg', reuse=True)
vgg_net1,_ = stylized_vgg16(img_input, pool='avg', reuse=True)


loss = tf.reduce_mean([tf.losses.mean_squared_error(vgg_src['conv3_3'],vgg_net['conv3_3'])]) + \
    tf.reduce_mean([tf.losses.mean_squared_error(vgg_src1['conv3_3'],vgg_net1['conv3_3']) ])


my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='reconstructed')
optimizer = tf.train.AdamOptimizer(0.1)
opt = optimizer.minimize(loss, var_list=my_vars)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

sess.run(tf.variables_initializer(optimizer.variables()+my_vars))


#tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
#for tensor in tensors:
#    print(tensor)

#conv1_w = tf.get_default_graph().get_tensor_by_name('stylized_vgg/conv1_2/conv2d/filter:0')
#conv1_w = vgg_src['conv2_2']

for i in tqdm(range(1000)):
    _, res, l = sess.run([opt, img_rec, loss], feed_dict={data:np.ones(shape=(1,patch_size,patch_size,3), dtype=np.float32), data_src:target_image})


    res = unprocess(res)


    if i % 50 == 0:
        print(i, l)
        r = np.clip(res[0],0,255).astype(np.uint8)
        put_img = Image.fromarray(r, mode='RGB').save('out1.png')









