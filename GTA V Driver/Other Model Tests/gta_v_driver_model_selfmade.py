import tensorflow as tf

from tensorflow.contrib import learn

width=640
height=160
learning_rate=0.0001

def gta_v_driver_model(features, mode):
    
    speed = features[:,0]
    input_layer = features[:,1:]

    #Input Layer
    input_layer = tf.reshape(input_layer, [-1, height, width, 3])
	
    #minimap = tf.image.crop_to_bounding_box(input_layer, 91, 90, 57, 89)
	
    conv1 = tf.layers.conv2d(
    	inputs=input_layer,
    	filters=32,
    	kernel_size=[13,13],
		strides=[2,2],
    	padding="valid",
    	activation=tf.nn.relu)
    
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2, padding='same')
    
    #Convolutional Layer #2
    conv2 = tf.layers.conv2d(
    	inputs=pool1,
    	filters=48,
    	kernel_size=[5,5],
		strides=[2,2],
    	padding="valid",
    	activation=tf.nn.relu)
    
    #Convolutional Layer #3
    conv3 = tf.layers.conv2d(
    	inputs=conv2,
    	filters=64,
    	kernel_size=[5,5],
		strides=[2,2],
    	padding="valid",
    	activation=tf.nn.relu)
    
    #Convolutional Layer #4
    conv4 = tf.layers.conv2d(
    	inputs=conv3,
    	filters=96,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    
    #Convolutional Layer #5
    conv5 = tf.layers.conv2d(
    	inputs=conv4,
    	filters=128,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
		
	#Convolutional Layer #6
    conv6 = tf.layers.conv2d(
    	inputs=conv5,
    	filters=256,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
	
    #Dense Layer
    conv6_flat = tf.contrib.layers.flatten(conv6)
	
	#Add additional feature to dense layers (speed)
    conv6_flat = tf.concat([tf.reshape(speed, [-1, 1]), conv6_flat], axis=1)
	
    print(conv6_flat.shape)
    
    dense1 = tf.layers.dense(inputs=conv6_flat, units=3072, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=3072, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    
    dense3 = tf.layers.dense(inputs=dropout2, units=3072, activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    
    dense4 = tf.layers.dense(inputs=dropout3, units=2048, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(inputs=dense4, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
	
    dense5 = tf.layers.dense(inputs=dropout4, units=1024, activation=tf.nn.relu)
    dropout5 = tf.layers.dropout(inputs=dense5, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    
    #Output
    predictions = tf.layers.dense(inputs=dropout5, units=3, activation=None)
	
    return predictions


