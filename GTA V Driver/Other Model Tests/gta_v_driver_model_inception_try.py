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
	
	conv1 = tf.layers.conv2d(
    	inputs=input_layer,
    	filters=24,
    	kernel_size=[7,7],
		strides=[2,2],
    	padding="valid",
    	activation=tf.nn.relu)
		
	conv2 = tf.layers.conv2d(
    	inputs=conv1,
    	filters=32,
    	kernel_size=[5,5],
		strides=[2,2],
    	padding="valid",
    	activation=tf.nn.relu)
		
	print(conv2.shape)
	
    inception_module_1 = inception_module(conv2, 24, 32)
	
    inception_module_2 = inception_module(inception_module_1, 32, 48)
	
    inception_module_3 = inception_module(inception_module_2, 48, 64)
	
	print(inception_module_3.shape)
	
	conv3 = tf.layers.conv2d(
    	inputs=inception_module_3,
    	filters=128,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
		
	print(conv3.shape)
		
	conv4 = tf.layers.conv2d(
    	inputs=conv3,
    	filters=192,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
		
	conv5 = tf.layers.conv2d(
    	inputs=conv4,
    	filters=256,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
	
    #Dense Layer
    conv5_flat = tf.contrib.layers.flatten(conv5)
	
	#Add additional feature to dense layers (speed)
    conv5_flat = tf.concat([tf.reshape(speed, [-1, 1]), conv5_flat], axis=1)
	
    print(conv5_flat.shape)
    
    dense1 = tf.layers.dense(inputs=conv5_flat, units=3072, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=3072, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense3 = tf.layers.dense(inputs=dropout2, units=2048, activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense4 = tf.layers.dense(inputs=dropout3, units=1024, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    #Output
    predictions = tf.layers.dense(inputs=dropout4, units=3, activation=None)
	
    return predictions
	
def inception_module(input, reduce, output_filters):
	
	#Follows input
	
	conv_1x1_1 = tf.layers.conv2d(
    	inputs=input,
    	filters=output_filters,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
		
	conv_1x1_2 = tf.layers.conv2d(
    	inputs=input,
    	filters=reduce,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
	
	conv_1x1_3 = tf.layers.conv2d(
    	inputs=input,
    	filters=reduce,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
		
	max_pool = tf.layers.max_pooling2d(
		inputs=input,
		pool_size=[3,3],
		strides=1,
		padding="same")
		
	#----------------------------------------------
		
	#Follows conv_1x1_2
	conv_3x3 = tf.layers.conv2d(
    	inputs=conv_1x1_2,
    	filters=output_filters,
    	kernel_size=[3,3],
    	padding="same",
    	activation=tf.nn.relu)
		
	#Follows conv_1x1_3
	conv_5x5 = tf.layers.conv2d(
    	inputs=conv_1x1_3,
    	filters=output_filters,
    	kernel_size=[5,5],
    	padding="same",
    	activation=tf.nn.relu)
		
	#Follows max_pool
	conv_1x1 = tf.layers.conv2d(
    	inputs=max_pool,
    	filters=output_filters,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
	
	return tf.concat([conv_1x1_1, conv_3x3, conv_5x5, conv_1x1], axis=3)


