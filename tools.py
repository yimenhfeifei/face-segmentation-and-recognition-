
import tensorflow as tf
import numpy as np
from scipy.io import loadmat

#%%
def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):

   
    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=is_pretrain,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=is_pretrain,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

#%%
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):

    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x

#%%
def batch_norm(x):

    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x


def FC_layer(layer_name, x, out_nodes):

    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x)
        return x


def loss(logits, labels):


    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
    loss = tf.reduce_mean(cross_entropy, name='loss')

    return loss
    

def accuracy(logits, labels):

  with tf.name_scope('accuracy') as scope:

      correct = tf.nn.in_top_k(logits, labels, 1)  
      correct = tf.cast(correct, tf.float16)  
      accuracy = tf.reduce_mean(correct)
#      compute the mean of correct([0,0,0,0,0,0,1,1,1,1,0,0,0,0,0])16 
      #accuracy=4/16
      tf.summary.scalar('/accuracy', accuracy)  
  return accuracy  
 




def num_correct_prediction(logits, labels):

#  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
#  correct = tf.cast(correct, tf.int32)
#  n_correct = tf.reduce_sum(correct)
#  #Computes the sum of elements across dimensions of a tensor. 
#  return n_correct
  correct = tf.nn.in_top_k(logits, labels, 1)  
  correct = tf.cast(correct, tf.float16)  
  accuracy = tf.reduce_mean(correct)
#      compute the mean of correct([0,0,0,0,0,0,1,1,1,1,0,0,0,0,0])16 
  tf.summary.scalar('/accuracy', accuracy)  
  return accuracy  



#%%
def optimize(loss, learning_rate, global_step):

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
        return train_op
    


    

          


def test_load():
    data_path = './/vgg-face.mat'
    x = loadmat(data_path)
    

    layers = x['layers']
##    current = input_maps
#    network = {}
    for layer in layers[0]:
        print("layer************\n",layer)
#        
        name = layer[0]['name'][0][0]
#        # conv1_1 / relu1_1/ conv1_2/ relu1_2/conv2_1/relu2_1/conv2_2/
        print("name/n",name)
        layer_type = layer[0]['type'][0][0]
#    
        if layer_type == 'conv':

              kernel,bias = layer[0]['weights'][0][0]
              
              
              print("kernel.shape\n",kernel.shape)
              print("*******************************\n")
              print("kernel\n",kernel)
              kernel = np.squeeze(kernel)
              print("***********************new kernel\n",kernel)
              print("***********************new kernel.shape\n",kernel.shape)
              print("*******************************\n")
              print("bias.shape\n",bias.shape)
              print("*******************************\n")
              print("bias/n",bias)


    
              


def load_with_skip(data_path, session, skip_layer):
  
    x = loadmat(data_path)
#    layers = x['layers']
    layers = x['layers']
    
    for layer in layers[0]:
        layer_type = layer[0]['type'][0][0]
        name = layer[0]['name'][0][0]
        if name not in skip_layer: 
             with tf.variable_scope(name, reuse=True):
              if layer_type == 'conv':
                 kernel,bias =layer[0]['weights'][0][0]
                 kernel = np.squeeze(kernel)
                 bias=np.squeeze(bias).reshape(-1)
                 session.run(tf.get_variable('weights').assign(kernel))
                 session.run(tf.get_variable('biases').assign(bias))
                     

                           
   
    
       
           

   





def weight(kernel_shape, is_uniform = True):

    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())    
    return w


def bias(bias_shape):
   
    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b











    
