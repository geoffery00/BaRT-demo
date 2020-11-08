import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import os
import sys
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from PIL import Image
slim = tf.contrib.slim
from nets import resnet_v2
from nets import vgg
from matplotlib import pyplot as plt
from preprocessing import vgg_preprocessing
from preprocessing import inception_preprocessing

#set image_dir
image_path = "/home/geo/bartdemo/BaRTDefense/images/"
test_image = "/home/geo/bartdemo/BaRTDefense/images/koala.jpg"
checkpoint_path = "./resnet_enhaced_checkpoint/"
image_size = resnet_v2.resnet_v2_50.default_image_size


with tf.Graph().as_default():
    
    image_contents = tf.read_file(test_image)
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    #resnet_v2_50 uses same preprocessing as vgg
    processed_image = vgg_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images = tf.expand_dims(processed_image, 0)
    
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        logits, _ = resnet_v2.resnet_v2_50(inputs = processed_images, 
                                           is_training=False, 
                                           num_classes=1001, 
                                           scope='resnet_v2_50', 
                                           reuse=tf.AUTO_REUSE)
    
    probabilities = tf.nn.softmax(logits)   
    init = tf.global_variables_initializer()
    saver = tf.train.import_meta_graph(checkpoint_path + 'resnet_enhanced.ckpt.meta')

    with tf.Session() as sess:
                
        sess.run(init)
        
        saver.restore(sess,tf.train.latest_checkpoint(checkpoint_path))
        
        np_image, probabilities = sess.run([image, probabilities])
        
        probabilities = probabilities[0, 0:]
        
        print("predicted output: \n")
        print(np.argmax(probabilities))