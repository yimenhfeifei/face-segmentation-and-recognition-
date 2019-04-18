




#%%
#import cv2
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
from FCN8s_keras import FCN
#model = FCN()
#model.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")
import cv2

#%%

def get_file(file_dir):
    '''Get full image directory and corresponding labels
    Args:
        file_dir: file directory
    Returns:
        images: image directories, list, string
        labels: label, list, int
    '''

    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        # image directories
        for name in files:
            
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))

            
    # assign 10 labels based on the folder names
    labels = []    
    i=0   
    for one_folder in temp:        
        n_img = len(os.listdir(one_folder))
        print("n_img is ",n_img)

        labels = np.append(labels, n_img*[i])
        i=i+1
    print("lables/n",labels)
        
#        labels_max = tf.reduce_max(labels)
#        print("lable-max/n",labels_max)
#    **********
#  cat and dog data input
#    cats = []  
#    label_cats = [] int64_feature 
#    dogs = []  
#    label_dogs = []  
#    for file in os.listdir(file_dir):  
#        name = file.split('.')  
#        if name[0]=='cat':  
#            cats.append(file_dir + file)  
#            label_cats.append(0)  
#        else:  
#            dogs.append(file_dir + file)  
#            label_dogs.append(1)  
#    print('There are %d cats\nThere are %d dogs' %(len(cats), len(dogs)))  
#      
#    image_list = np.hstack((cats, dogs))  
#    label_list = np.hstack((label_cats, label_dogs))  
#      
#    temp = np.array([image_list, label_list])  
#    temp = temp.transpose()  
#    np.random.shuffle(temp)  
#      
#    image_list = list(temp[:, 0])  
#    label_list = list(temp[:, 1])  
#    label_list = [int(i) for i in label_list]  
#      
#      
#    return image_list, label_list  
#  *******
    temp = np.array([images, labels])
#    print("*****************temp is \n ",temp)
    temp = temp.transpose()
    print("*************temptrans pose is \n",temp)
    np.random.shuffle(temp)
    print("*************temptrans shuffle is \n",temp)

    
    image_list = list(temp[: , 0])
    
#    print(image_list)
#    print("*************image_list  is \n",image_list)
    label_list = list(temp[:, 1])
#    print("*************label_list  is \n",label_list)
    label_list = [int(float(i)) for i in label_list]
      
    return image_list, label_list
def get_batch(image, label, image_W, image_H, batch_size, capacity,shuffle):
    
    
#    print(image)
#    image = tf.cast(image, tf.string) 
##    print(image.shape)
##    label = tf.cast(label, tf.int32) 
    images = []
    temp = []
#    print("max lable is ",label)
#    n_samples = len(label)
##   
##    
#    for i in np.arange(0, n_samples):
#          image[i]=Image.open(image[i])
#          image[i]= image[i].resize((500,500))
#          
##          print("image.dtype",image.dtype) 
##          print("image[i].shape",image[i].size) 
#          plt.imshow(image[i])
#          b=image[i]
##          print(b)
##          b=b.split(sep='/')
##          b=b[9]
##          print("b",b)home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/data/data2/training/' 
#          image[i] = cv2.cvtColor(cv2.imread(image[i]),cv2.COLOR_BGR2RGB)
#          
#          
#          a=FCN(image[i])
##          
####          print("seg_image",seg_image) 
##          images.append(seg_image)
#          print("images",images)
#          image[i] = tf.convert_to_tensor(image[i], dtype=tf.float32)
#          image[i]=tf.squeeze(image[i])
#          
###          image[i] = tf.image.resize_image_with_crop_or_pad(image[i], image_W, image_H)
##       
#          image[i] = np.asarray(image[i])
#          path1 = '/home/hadoop/Desktop/bbb/aaa/'
#          filename1=path1+'%s'%(b)+'%s.jpg'%(i)
#          cv2.imwrite(os.path.join(filename1), a)

              
              

     
        
#    for root, sub_folders, files in os.walk(path1):
#        # image directories
#        for name in files:
#            images.append(os.path.join(root, name))
#            print(images)
#    temp = np.array([images,label])
#    np.random.shuffle(temp)
#    image = list(temp[: , 0])
#    print("image",image)
        



    image = tf.cast(image, tf.string) 

    label = tf.cast(label, tf.int32) 
     
    input_queue = tf.train.slice_input_producer([image, label])
#    print("input_queue",input_queue)
#    
    label = input_queue[1]
    
    
    
    image_contents = tf.read_file(input_queue[0])
#    print("image_contents.shape",input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
#    print("image.shape",image)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    
#    randomly crop the image size to 24 x 24
    image = tf.image.flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image,lower=0.2,upper=1.8)

    image = tf.image.per_image_standardization(image)
   
    if shuffle:
   
    
      image_batch, label_batch = tf.train.shuffle_batch([image,label],
                                                      batch_size=batch_size,
                                                      num_threads=4,
                                                      capacity=capacity,
                                                      min_after_dequeue=100)
    else:
       image_batch, label_batch = tf.train.batch([image,label],
                                                      batch_size=batch_size,
                                                      num_threads=4,
                                                      capacity=capacity,)
        
    print("image_batch",image_batch.shape)
    label_batch = tf.reshape(label_batch, [batch_size])
#    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch,label_batch

#%%

#def int64_feature(value):
#  """Wrapper for inserting int64 features into Example proto."""
#  if not isinstance(value, list):
#    value = [value]
#  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#
#def bytes_feature(value):
#  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#%%

#def convert_to_tfrecord(images, labels, save_dir, name):
#    '''convert all images and labels to one tfrecord file.
#    Args:
#        images: list of image directories, string type image_batch, label_batch
#        labels: list of labels, int type
#        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
#        name: the name of tfrecord file, string type, e.g.: 'train'
#    Return:
#        no return
#    Note:
#        converting needs some time, be patient...
#    '''
#    
#    filename = os.path.join(save_dir, name + '.tfrecords')
#    n_samples = len(labels)
#    
#    if np.shape(images)[0] != n_samples:
#        raise ValueError('Images size %d does not match label size %d.' %(images.shape[0], n_samples))
    
    
    
    # wait some time here, transforming need some time based on the size of your data.
#    writer = tf.python_io.TFRecordWriter(filename)
#    print('\nTransform start......')
#    for i in np.arange(0, n_samples):
#        try:
#            img=Image.open(images[i])
#            img= img.resize((196,196))
##            image = io.imread(images[i])
#             # type(image) must be array!
##            print(img.size)
#            image_raw = img.tostring()
#            label = int(labels[i])
#            image = cv2.imread(images[i])  
#            seg_image=fcn(images[i])
##            image = cv2.resize(image, (196, 196)) 
#            image = cv2.resize(seg_image, (500, 500)) 
#            print("image weight+height",image.shape)
#          
#           
##            b,g,r = cv2.split(image)  
##            rgb_image = cv2.merge([r,g,b])
#            image_raw =  image.tostring()
#            label = int(labels[i])
#            example = tf.train.Example(features=tf.train.Features(feature={
#                            'label':int64_feature(label),
#                            'image_raw': bytes_feature(image_raw)}))
#            writer.write(example.SerializeToString())
#        except IOError as e:
#            print('Could not read:', images[i])
#            print('error: %s' %e)
#            print('Skip it!\n')
#    writer.close()
#    print('Transform done!')
    

##%%
#
#def read_and_decode(tfrecords_file,batch_size):
#    '''read and decode tfrecord file, generate (image, label) batches
#    Args:
#        tfrecords_file: the directory of tfrecord file
#        batch_size: number of images in each batch
#    Returns:
#        image: 4D tensor - [batch_size, width, height, channel]
#        label: 1D tensor - [batch_size]
#    '''
#    # make an input queue from the tfrecord file
#    filename_queue = tf.train.string_input_producer([tfrecords_file])
#    
#    reader = tf.TFRecordReader()
#    _, serialized_example = reader.read(filename_queue)
#    img_features = tf.parse_single_example(
#                                        serialized_example,
#                                        features={
#                                               'label': tf.FixedLenFeature([], tf.int64),
#                                               'image_raw': tf.FixedLenFeature([], tf.string) 
#                                               })
#    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
##    image = tf.decode_raw(img_features['image_raw'], tf.float32)
#    
#    print("hahahahha",image.shape)
#    
#    ##########################################################
#    # you can put data augmentation here, I didn't use it
#    ##########################################################
#    # all the images of notMNIST are 28*28, you need to change the image size if you use other dataset.
#    
#
#    image = tf.reshape(image, [500,500,3])
##    image = tf.reshape(image, [180,200,3])
#    a,b, c = image.shape
#    print("a,b,c",a,b,c)
#    #int32
#    label = tf.cast(img_features['label'], tf.int64) 
##    shape = tf.cast(img_features['shape'], tf.int32)  
#    image_batch, label_batch = tf.train.batch([image, label],
#                                                batch_size= batch_size,
#                                                num_threads= 64, 
#                                                capacity = 2000)
##                                                min_after_dequeue=1000)
##    print(label_batch.shape)
##    print(tf.reshape(label_batch, [batch_size]))
#
#    return image_batch, tf.reshape(label_batch, [batch_size])
##    return image, label
#
#def vgg_preprocess(im):
#    im = cv2.resize(im, (500, 500))
#    print("im.shape",im.shape)
#    in_ = np.array(im, dtype=np.float32)
#    in_ = in_[:,:,::-1]
#    in_ -= np.array((104.00698793,116.66876762,122.67891434))
#    in_ = in_[np.newaxis,:]
    #in_ = in_.transpose((2,0,1))
#    return in_
#def fcn(im):
#    
##    print("im.shape",im.shape)
###    im=im[:,:,0]
##    print("im.shape",im.shape)
##    print("type",im.dtype)
##    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#    inp_im = vgg_preprocess(im)
#    print("im.shape",im.shape)
#    out = model.predict([inp_im])
##    out_resized = cv2.resize(np.squeeze(out), (500,500))
#    out_resized = cv2.resize(np.squeeze(out), (im.shape[1],im.shape[0]))
#    out_resized_clipped = np.clip(out_resized.argmax(axis=2), 0, 1).astype(np.float64)
#    mask = cv2.GaussianBlur(out_resized_clipped, (7,7), 4) 
##    
##    print("mask weight+height",mask.shape)
#    a=mask[:,:,np.newaxis]
#    b=(a*im.astype(np.float64)).astype(np.uint8)
#    print("b.shape",b.shape)
##    plt.imshow((mask[:,:,np.newaxis]*im.astype(np.float64)).astype(np.uint8))
#    return b
  
#%% Convert data to TFRecord
#test_dir = './/Data/notMNIST_small//'
#test_dir = '/home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/data/data/dataseg/'
#save_dir = ''
#BATCH_SIZE = 16
###196
###
####Convert test data: you just need to run it ONCE !
#name_test = 'data_seg'
#images, labels = get_file(test_dir)
############
##############################################
##############a= len(images)
##############new_seg_image=[]
##############for i in range(a):
##############    seg_image=fcn(images[i])
##############    
##############    new_seg_image = np.append(new_seg_image, [seg_image])
###############    
#############
#############
###############################################
#############
#########labels_max = tf.reduce_max(labels)
###########
#########sess = tf.Session()
#########print("lables size",sess.run(labels_max))
##########
#########
#convert_to_tfrecord(images, labels, save_dir, name_test)
##
#import VGG
#import tools
#def train():
#
#    learning_rate=0.001
#    tfrecords_file = './/new_test.tfrecords'
#    image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=3)
#    image_batch = tf.cast(image_batch,dtype=tf.float32)
#    label_batch = tf.cast(label_batch,dtype=tf.int64)
#    print(image_batch)
#    print(label_batch)
#    IS_PRETRAIN=True
#    logits = VGG.VGG16N(image_batch, 2, IS_PRETRAIN)
#    print("logits",logits)#import VGG
##import tools
#def train():
#
#    learning_rate=0.001
#    tfrecords_file = './/new_test.tfrecords'
#    image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=3)
#    image_batch = tf.cast(image_batch,dtype=tf.float32)
#    label_batch = tf.cast(label_batch,dtyp image = tf.cast(image, tf.string) e=tf.int64)
#    print(image_batch)
#    print(label_batch)
#    IS_PRETRAIN=True
#    logits = VGG.VGG16N(image_batch, 2, IS_PRETRAIN)
#    print("logits",logits)
#    loss = tools.loss(logits, label_batch)
#    print("loss",loss)
#    accuracy = tools.accuracy(logits, label_batch)
#    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
#    train_op = tools.optimize(loss, learning_rate, my_global_step)   
#    init = tf.global_variables_initializer()
#    sess = tf.Session()
#    sess.run(init)
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
#    
#    try:
#         for step in np#tfrecords_file = './/test.tfrecords'
#image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
#print(image_batch.shape)
##with tf.Session()  as sess:
#    
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            # just plot one batch size            
##            image, label = sess.run([image_batch, label_batch])
##            print(image.shape)
##            print("lable is ",label.shape)
##            plot_images(image, label)
##            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#    coord.join(threads).arange(10):
#            if coord.should_stop():
#              break
#            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy])  
#            if step % 1 == 0 or (step + 1) == 10: 
#                  print ('Step: %d, loss: %.2f, accuracy: %.2f%%' % (step, tra_loss, tra_acc*100))
#    except tf.errors.OutOfRangeError:
#             print('Done training -- epoch limit reached')
#    finally:
#        coord.request_stop()
#        
#    coord.join(threads)
#    sess.close()
#    loss = tools.loss(logits, label_batch)
#    print("loss",loss)
#    accuracy = tools.accuracy(logits, label_batch)
#    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
#    train_op = tools.optimize(loss, learning_rate, my_global_step)   
#    init = tf.global_variables_initializer()
#    sess = tf.Session()
#    sess.run(init)
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  
#    
#    try:
#         for step in np.arange(10):
#            if coord.should_stop():
#              break
#            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy])  
#            if step % 1 == 0 or (step + 1) == 10: 
#                  print ('Step: %d, loss: %.2f, accuracy: %.2f%%' % (step, tra_loss, tra_acc*100))
#    except tf.errors.OutOfRangeError:
#             print('Done training -- epoch limit reached')
#    finally:
#        coord.request_stop()
#        
#    coord.join(threads)
#    sess.close()


#%% TO test train.tfrecord file
#BATCH_SIZE=6   
#def plot_images(images, labels):
#    '''plot one batch size
#    '''
#    for i in np.arange(0, 16):
#             plt.subplot(4, 4, i + 1)
#             plt.axis('off')
##             plt.figure(figsize=(5,5))
##             plt.title(chr(ord('A') + labels[i] - 1), fontsize = 14)
#             plt.subplots_adjust(top=1.5)
#            
#             plt.imshow(images[i])
#             
#     
##    _, axs = plt.subplots(2, 10, figsize=(12, 12))
##   
##    for i in range(20):
##        for img, ax in zip(images[i], axs):
##               ax.imshow(img)
##               plt.imshow()
#
#    
#data_dir='/home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/data/segmentation/test/'
#image, label=get_file(data_dir)
#image_batch,label_batch=get_batch(image, label, 800, 800,16, 1000,True)
##tfrecords_file = './/data_seg.tfrecords'
##image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
##print(image_batch.shape)
#with tf.Session()  as sess:
#    
#    i = 0
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    try:
#        while not coord.should_stop() and i<1:
#            # just plot one batch size            
#            image, label = sess.run([image_batch, label_batch])
#            print("image.shape",image.shape)
##            print(image.shape)
##            print("lable is ",label.shape)
#            plot_images(image, label)
#            i+=1
#            
#    except tf.errors.OutOfRangeError:
#        print('done!')
#    finally:
#        coord.request_stop()
#        coord.join(threads)

#     image_path='.//Angela Bassett1141.jpg'
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(cv2.imread(image[i]),cv2.COLOR_BGR2RGB)
#     cv2.imshow('BGR Image',img )    
