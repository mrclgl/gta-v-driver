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
    
	# Convolutional Layer #1
    conv1 = tf.layers.conv2d(
    	inputs=input_layer,
    	filters=96,
    	kernel_size=[11,11],
		strides=[4,4],
    	padding="valid",
    	activation=tf.nn.relu)
    
    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=2, padding='same')
    
    #Convolutional Layer #2
    conv2 = tf.layers.conv2d(
    	inputs=pool1,
    	filters=256,
    	kernel_size=[5,5],
    	padding="valid",
    	activation=tf.nn.relu)
    
    #Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=2, padding='same')
    
    #Convolutional Layer #3
    conv3 = tf.layers.conv2d(
    	inputs=pool2,
    	filters=384,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    
    #Convolutional Layer #4
    conv4 = tf.layers.conv2d(
    	inputs=conv3,
    	filters=384,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    
    #Convolutional Layer #5
    conv5 = tf.layers.conv2d(
    	inputs=conv4,
    	filters=256,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
		
	#Pooling Layer #3
    pool3 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3,3], strides=2, padding='same')
	
    #Dense Layer
    pool3_flat = tf.contrib.layers.flatten(pool3)
	
	#Add additional feature to dense layers (speed)
    pool3_flat = tf.concat([tf.reshape(speed, [-1, 1]), pool3_flat], axis=1)
	
    print(pool3_flat.shape)
    
    dense1 = tf.layers.dense(inputs=pool3_flat, units=4096, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    #Output
    predictions = tf.layers.dense(inputs=dropout2, units=3, activation=None)
	
    return predictions


