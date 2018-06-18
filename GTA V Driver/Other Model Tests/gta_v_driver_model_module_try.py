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
    
    tf.summary.image("0_Input_Layer", tf.expand_dims(input_layer[0], 0))
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
    	inputs=input_layer,
    	filters=48,
    	kernel_size=[7,7],
    	strides=[2,2],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("1_Conv1_Layer", tf.transpose(tf.expand_dims(conv1[0], 0), [3, 1, 2, 0]), max_outputs=5)
    	
    # Convolutional Layer #2
    conv2 = tf.layers.conv2d(
    	inputs=conv1,
    	filters=64,
    	kernel_size=[7,7],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("2_Conv2_Layer", tf.transpose(tf.expand_dims(conv2[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    module1 = module(conv2, 32, 64)
    
    tf.summary.image("3_Module1", tf.transpose(tf.expand_dims(module1[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    module2 = module(module1, 32, 64)
    
    tf.summary.image("4_Module2", tf.transpose(tf.expand_dims(module2[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    module3 = module(module2, 32, 64)
    
    tf.summary.image("5_Module3", tf.transpose(tf.expand_dims(module3[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    module4 = module(module3, 32, 64)
    
    tf.summary.image("6_Module4", tf.transpose(tf.expand_dims(module4[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    module5 = module(module4, 32, 64)
    
    tf.summary.image("7_Module5", tf.transpose(tf.expand_dims(module5[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    module6 = module(module5, 32, 64)
    
    tf.summary.image("8_Module6", tf.transpose(tf.expand_dims(module6[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    # Convolutional Layer #3
    conv3 = tf.layers.conv2d(
    	inputs=module6,
    	filters=64,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
    	
    tf.summary.image("9_Conv3_Layer", tf.transpose(tf.expand_dims(conv3[0], 0), [3, 1, 2, 0]), max_outputs=5)
	
	#Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3,3], strides=2, padding='same')
    
    # Convolutional Layer #4
    conv4 = tf.layers.conv2d(
    	inputs=pool1,
    	filters=96,
    	kernel_size=[5,5],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("10_Conv4_Layer", tf.transpose(tf.expand_dims(conv4[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    # Convolutional Layer #5
    conv5 = tf.layers.conv2d(
    	inputs=conv4,
    	filters=128,
    	kernel_size=[5,5],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("11_Conv5_Layer", tf.transpose(tf.expand_dims(conv5[0], 0), [3, 1, 2, 0]), max_outputs=5)
	
	#Pooling Layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3,3], strides=2, padding='same')
    
    # Convolutional Layer #6
    conv6 = tf.layers.conv2d(
    	inputs=pool2,
    	filters=64,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
    	
    tf.summary.image("12_Conv6_Layer", tf.transpose(tf.expand_dims(conv6[0], 0), [3, 1, 2, 0]), max_outputs=5)
	
	# Convolutional Layer #7
    conv7 = tf.layers.conv2d(
    	inputs=conv6,
    	filters=96,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("13_Conv7_Layer", tf.transpose(tf.expand_dims(conv7[0], 0), [3, 1, 2, 0]), max_outputs=5)
	
	# Convolutional Layer #8
    conv8 = tf.layers.conv2d(
    	inputs=conv7,
    	filters=128,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("14_Conv8_Layer", tf.transpose(tf.expand_dims(conv8[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
	# Convolutional Layer #9
    conv9 = tf.layers.conv2d(
    	inputs=conv8,
    	filters=64,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
    	
    tf.summary.image("15_Conv9_Layer", tf.transpose(tf.expand_dims(conv9[0], 0), [3, 1, 2, 0]), max_outputs=5)
	
    print(conv9.shape)
    
    #Dense Layer
    conv9_flat = tf.contrib.layers.flatten(conv9)
    
    #Add additional feature to dense layers (speed)
    conv9_flat = tf.concat([tf.reshape(speed, [-1, 1]), conv9_flat], axis=1)
    
    print(conv9_flat.shape)
    
    dense1 = tf.layers.dense(inputs=conv9_flat, units=4096, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense3 = tf.layers.dense(inputs=dropout2, units=3072, activation=tf.nn.relu)
    dropout3 = tf.layers.dropout(inputs=dense3, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense4 = tf.layers.dense(inputs=dropout3, units=2048, activation=tf.nn.relu)
    dropout4 = tf.layers.dropout(inputs=dense4, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    dense5 = tf.layers.dense(inputs=dropout4, units=1024, activation=tf.nn.relu)
    dropout5 = tf.layers.dropout(inputs=dense5, rate=0.5, training=mode == learn.ModeKeys.TRAIN)
    
    #Output
    predictions = tf.layers.dense(inputs=dropout5, units=3, activation=None)
    
    return predictions
    
def module(input, reduce, output_filters):
	
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
		pool_size=[2,2],
		strides=1,
		padding="same")
		
	#----------------------------------------------
		
	#Follows conv_1x1_2
	conv_5x5 = tf.layers.conv2d(
    	inputs=conv_1x1_2,
    	filters=output_filters,
    	kernel_size=[5,5],
    	padding="same",
    	activation=tf.nn.relu)
		
	#Follows conv_1x1_3
	conv_3x3 = tf.layers.conv2d(
    	inputs=conv_1x1_3,
    	filters=output_filters,
    	kernel_size=[3,3],
    	padding="same",
    	activation=tf.nn.relu)
		
	#Follows max_pool
	conv_1x1 = tf.layers.conv2d(
    	inputs=max_pool,
    	filters=output_filters,
    	kernel_size=[1,1],
    	padding="same",
    	activation=tf.nn.relu)
	
	return tf.concat([conv_1x1_1, conv_5x5, conv_3x3, conv_1x1], axis=3)
    
    