


    
#%%

import os
import os.path

import numpy as np
import tensorflow as tf
import time
#import input_data
import VGG
import tools
import notMNIST_input
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#%%
IMG_W = 300
IMG_H = 300
#N_CLASSES = 85
N_CLASSES = 85
#N_CLASSES_vali = 40
BATCH_SIZE = 5
starter_learning_rate = 0.001
MAX_STEP = 10000 # it took me about one hour to complete the training.
IS_PRETRAIN = True
capacity=400


#%%   Training
#validate_file='.//validate_test.tfrecords'
#tfrecords_file = './/training_test.tfrecords'
#def get_dataset(fname):  
#    dataset = tf.data.TFRecordDataset(fname)  
#    return dataset.map(notMNIST_input.read_and_decode) 
   
  
def train():
    
#    pre_trained_weights1 = './/vgg16.npy'
    pre_trained_weights = './/vgg-face.mat'
    data_dir = '/home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/data/segmentation/training/'   
    train_log_dir = './/logss/train_shuffle/'
    val_log_dir = './/logss/va_shuffle/'
    

   
#    image_batch, label_batch = notMNIST_input.read_and_decode(tfrecords_file,BATCH_SIZE)
    image, label=notMNIST_input.get_file(data_dir)
#        image_batch,label_batch=notMNIST_input.get_batch(image, label, IMG_W, IMG_H, BATCH_SIZE, capacity)
    X=np.array(image)
    Y=np.array(label)
    kf = KFold(n_splits=10,shuffle=False)
    total_acc=0
    for train, test in kf.split(X,Y):
        tf.reset_default_graph()
        image_batch,label_batch=notMNIST_input.get_batch(X[train], Y[train], IMG_W, IMG_H, BATCH_SIZE, capacity,shuffle=True)
        image_batch_validate, label_batch_validate=notMNIST_input.get_batch(X[test], Y[test], IMG_W, IMG_H, BATCH_SIZE, capacity,shuffle=False)
#        print("dddd")
##        print("train_index: , test_index:", (X[train],Y[train],X[test],Y[test]))
        print("X[train]/n",len(X[train]))
        print("Y[train]/n",len(Y[train]))
        print("X[test]",len(X[test]))
        print("Y[test]",len(Y[test]))

    
    #cast (1.8,3.4)float32 to (1,3)int64
     
        x=tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMG_W,IMG_H,3],name='place_x')
        y_=tf.placeholder(tf.int64,shape=[BATCH_SIZE,],name='place_y')  
        logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
        print("****logits shape is ",logits.shape)

        loss = tools.loss(logits, y_)

        print("label_batch is ",y_.shape)
        accuracy = tools.accuracy(logits, y_)

    
        my_global_step = tf.Variable(0, name='global_step', trainable=False) 
        #learning_rate = tf.train.exponential_decay(starter_learning_rate, my_global_step,
                                         #  2200, 0.96, staircase=True)
        train_op = tools.optimize(loss, starter_learning_rate, my_global_step)   
#    train_op_vali = tools.optimize(loss_vali, learning_rate, my_global_step)
    
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
    
        init = tf.global_variables_initializer()
        
        sess = tf.Session()
        
        sess.run(init)
        
    # load the parameter file, assign the parameters, skip the specific layers
        tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8']) 


        merged_summaries =tf.summary.merge_all()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)    
        tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir)
        max_acc = 0
        total_time=0
       
    
        try:
            for step in np.arange(MAX_STEP):
                   if coord.should_stop():
                                break
                   start_time = time.time()
#        with tf.Session() as sess:
                    

                      
#                 for train, test in kf.split(X,Y):
#                     image_batch,label_batch=notMNIST_input.get_batch(X[train], Y[train], IMG_W, IMG_H, BATCH_SIZE, capacity)
#                     image_batch_validate, label_batch_validate=notMNIST_input.get_batch(X[test], Y[test], IMG_W, IMG_H, BATCH_SIZE, capacity)
#                     label_batch = tf.cast(label_batch,dtype=tf.int64)
                   x_train_a, y_train_a = sess.run([image_batch, label_batch])
                   x_test_a,y_test_a=sess.run([image_batch_validate,label_batch_validate])         
#            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy]) 
#            tra_images,tra_labels = sess.run([image_batch, label_batch])
                   _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x:x_train_a, y_:y_train_a}) 
            
                   if step % 10 == 0 or (step + 1) == MAX_STEP:  
                        feed_dict={x:x_train_a, y_:y_train_a}
                        summary_str = sess.run(summary_op,feed_dict=feed_dict)
                        tra_summary_writer.add_summary(summary_str, step)
                        time_elapsed = time.time()-start_time
                        print ('Step:%d , loss: %.2f, accuracy: %.2f%%(%.2f sec/step)' % (step,tra_loss, tra_acc*100,time_elapsed))
                 
                        total_time=total_time+time_elapsed
                        if step%50==0: 
                            print('total time is :%.2f'%(total_time))
                        

                   if step % 200 == 0 or (step + 1) == MAX_STEP:
                
                     val_loss, val_acc = sess.run([loss, accuracy],
                                                  feed_dict={x:x_test_a, y_:y_test_a})
                     feed_dict={x:x_test_a, y_:y_test_a}
                     summary_str = sess.run(summary_op,feed_dict=feed_dict)
                     val_summary_writer.add_summary(summary_str, step)
                     

#                if cur_val_loss > max_acc:  
#                         max_acc = cur_val_loss  
#                         best_step = step 
#                         checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
#                         saver.save(sess, checkpoint_path, global_step=step)
#                val_summary_writer.add_summary(summary, step)  
#                print("Model updated and saved in file: %s" % checkpoint_path)
#                print ('*************step %5d: loss %.5f, acc %.5f --- loss val %0.5f, acc val %.5f************'%(best_step,tra_loss, tra_acc, cur_val_loss, cur_val_eval))

#                
                 
                     print ('************validate result:Step:%d , loss: %.2f, accuracy: %.2f%%(%.2f sec/step)' % (step,val_loss, val_acc*100,time_elapsed))                
                     if val_acc > max_acc:
                       max_acc = val_acc
                       checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                       saver.save(sess, checkpoint_path,global_step=step)
            if max_acc>total_acc:
               total_acc=max_acc
               checkpoint_path = os.path.join(val_log_dir, 'model.ckpt')
               saver.save(sess, checkpoint_path,global_step=step)
            
                
        except tf.errors.OutOfRangeError:
           print('Done training -- epoch limit reached')
        finally:
           coord.request_stop()
          
        coord.join(threads)
        sess.close()
    
        
    
 
    

train() 


    
#%%   Test the accuracy on test dataset. got about 85.69% accuracy.
import math

from PIL import Image
import matplotlib.pyplot as plt 

def get_one_image(train,lable):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    print(ind)
    img_dir = train[ind]
    lable_img=lable[ind]
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([200, 200])
    image = np.array(image)
    return image,lable_img 
def evaluate():
  with tf.Graph().as_default():
    BATCH_SIZE = 5
    N_CLASSES = 146
    log_dir = '/home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/logss/va_shuffle/'
    #log_dir = '/home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/logs/train/'
    #data_dir2 = '/home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/data/segmentation/test/'
    data_dir2='/home/hadoop/Desktop/My-TensorFlow-tutorials-master/VGG face segmentation  recognition/data/data1/testold/'

    image, label=notMNIST_input.get_file(data_dir2)
    image_batch,label_batch=notMNIST_input.get_batch(image, label, IMG_W, IMG_H, BATCH_SIZE, capacity,shuffle=False) 
    x=tf.placeholder(tf.float32,shape=[5,IMG_W,IMG_H,3],name='place_x')
    y_=tf.placeholder(tf.int64,shape=[5,],name='place_y')     
    image_batch = tf.cast(image_batch,dtype=tf.float32)
    label_batch = tf.cast(label_batch,dtype=tf.int64)

    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)   
   
    accuracy = tools.num_correct_prediction(logits, y_)
    saver = tf.train.Saver(tf.global_variables())
        
    with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                global_step=4500
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess = sess, coord = coord)
            
            try:
                print('\nEvaluating......')
                num_step = int(math.floor(len(image) / BATCH_SIZE))
#                num_sample = num_step*BATCH_SIZE
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
#                    print(step)
                    x_train_a, y_train_a = sess.run([image_batch, label_batch]) 
                    batch_correct = sess.run([accuracy],feed_dict={x:x_train_a,y_: y_train_a})
                    print("batch_correct:",batch_correct)
                    total_correct += np.sum(batch_correct)
                    print("total_correct:",total_correct)
                    step += 1
#                print('Total testing samples: %d' %num_sample)
                print('Total correct predictions: %d' %total_correct)
                print('Average accuracy: %.2f%%' %(100*total_correct/num_step))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
                
                
#evaluate()
