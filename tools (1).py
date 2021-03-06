
import tensorflow as tf
import numpy as np
from scipy.io import loadmat

#%%
def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=True):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''
   
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
        x = batch_norm(x) 
        x = tf.nn.relu(x, name='relu')
        return x

#%%
def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
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

#%%
def FC_layer(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
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

#%%
def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    lamda=0.01
    with tf.name_scope('loss') as scope:
       #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
#        loss = tf.reduce_mean(cross_entropy, name='loss')//lable is one-hot
#        tf.summary.scalar(scope+'/loss', loss)
       cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy_per_example')# lable is int
       loss = tf.reduce_mean(cross_entropy, name='loss')
#       print("loss1",loss)
       total_vars = tf.trainable_variables()
#       print("total_vars",total_vars)
#       weights_name_list = [var for var in total_vars if 'bias' not in var.name ]
#       loss_holder = []
#       for w in range(len(weights_name_list)):
#            l2_loss = tf.nn.l2_loss(weights_name_list[w])
#            loss_holder.append(l2_loss)
#       regular_loss = tf.reduce_mean(loss_holder)*lamda
#       loss = loss + regular_loss
       print("loss2",loss)
       tf.summary.scalar(scope+'/loss', loss)
    return loss
    
#%%
def accuracy(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, 
  """
  with tf.name_scope('accuracy') as scope:
#      correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
#      correct = tf.cast(correct, tf.float32)
#      accuracy = tf.reduce_mean(correct)*100.0
#      tf.summary.scalar(scope+'/accuracy', accuracy)
#      max_index = np.argmax(logits)
      print("logits is ", logits)
      n=np.array(labels)
      print("labels is ", n)
      correct = tf.nn.in_top_k(logits, labels, 1)  
      correct = tf.cast(correct, tf.float16)  
      accuracy = tf.reduce_mean(correct)
#      compute the mean of correct([0,0,0,0,0,0,1,1,1,1,0,0,0,0,0])16 
      #accuracy=4/16
      tf.summary.scalar(scope+'/accuracy', accuracy)  
  return accuracy  
 



#%%
def num_correct_prediction(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Return:
      the number of correct predictions
  """
#  correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
#  correct = tf.cast(correct, tf.int32)
#  n_correct = tf.reduce_sum(correct)
#  #Computes the sum of elements across dimensions of a tensor. 
#  return n_correct
  correct = tf.nn.in_top_k(logits, labels, 1)  
  correct = tf.cast(correct, tf.float16)  
  accuracy = tf.reduce_mean(correct)
#      compute the mean of correct([0,0,0,0,0,0,1,1,1,1,0,0,0,0,0])16 
  tf.summary.scalar('/num_correct_prediction', accuracy)  
  return accuracy  



#%%
def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
       # optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
        return train_op
    


    
#%%
          

#%%  
def test_load():
    data_path = './/vgg-face.mat'
    x = loadmat(data_path)
    

    layers = x['layers']
##    current = input_maps
#    network = {}
    for layer in layers[0]:
#        print("layer************\n",layer)
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
#              print("kernel\n",kernel)
              #kernel = /home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/data1/training/np.squeeze(kernel)
#              print("***********************new kernel\n",kernel)
              print("***********************new kernel.shape\n",kernel.shape)
              print("*******************************\n")
              print("bias.shape\n",bias.shape)
              print("*******************************\n")
#              print("bias/n",bias)


    
#%%                


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
#               kernel, bias =layer[0]['weights'][0][0]
#               kernel = np.squeeze(kernel)
#               bias = np.squeeze(bias).reshape(-1)
#               for b in bias:
                 kernel,bias =layer[0]['weights'][0][0]
#                 for w in weight:
                 kernel = np.squeeze(kernel)
#                 print("*********************name\n",name)   
#                 print("**********************K\n",kernel)
                 bias=np.squeeze(bias).reshape(-1)
#                 print("***********************b\n",bias)
#                 for subkey, data in zip(('weights'),kernel):
                 session.run(tf.get_variable('weights').assign(kernel))
                 session.run(tf.get_variable('biases').assign(bias))



                     

                           
   
    
       
           

   




#%%
def weight(kernel_shape, is_uniform = True):
    ''' weight initializer
    Args:
        shape: the shape of weight
        is_uniform: boolen type.
                if True: use uniform distribution initializer
                if False: use normal distribution initizalizer
    Returns:
        weight tensor
    '''
    w = tf.get_variable(name='weights',
                        shape=kernel_shape,
                        initializer=tf.contrib.layers.xavier_initializer())    
    return w

#%%
def bias(bias_shape):
    '''bias initializer
    '''
    b = tf.get_variable(name='biases',
                        shape=bias_shape,
                        initializer=tf.constant_initializer(0.0))
    return b

#%%









    