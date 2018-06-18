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
    
    tf.summary.image("Input_Layer", tf.expand_dims(input_layer[0], 0))
    
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
    	inputs=input_layer,
    	filters=48,
    	kernel_size=[7,7],
    	strides=[2,2],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("Conv1_Layer", tf.transpose(tf.expand_dims(conv1[0], 0), [3, 1, 2, 0]), max_outputs=5)
    	
    #Convolutional Layer #2
    conv2 = tf.layers.conv2d(
    	inputs=conv1,
    	filters=64,
    	kernel_size=[7,7],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    #Pooling Layer #1
    pool1 = tf.layers.average_pooling2d(inputs=conv2, pool_size=[2,2], strides=2, padding='same')
    
    tf.summary.image("Conv2_Layer", tf.transpose(tf.expand_dims(pool1[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    #Convolutional Layer #3
    conv3 = tf.layers.conv2d(
    	inputs=pool1,
    	filters=96,
    	kernel_size=[5,5],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("Conv3_Layer", tf.transpose(tf.expand_dims(conv3[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    #Convolutional Layer #4
    conv4 = tf.layers.conv2d(
    	inputs=conv3,
    	filters=128,
    	kernel_size=[5,5],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    #Pooling Layer #2
    pool2 = tf.layers.average_pooling2d(inputs=conv4, pool_size=[2,2], strides=2, padding='same')
    	
    tf.summary.image("Conv4_Layer", tf.transpose(tf.expand_dims(pool2[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    #Convolutional Layer #5
    conv5 = tf.layers.conv2d(
    	inputs=pool2,
    	filters=192,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    		
    tf.summary.image("Conv5_Layer", tf.transpose(tf.expand_dims(conv5[0], 0), [3, 1, 2, 0]), max_outputs=5)
    	
    #Convolutional Layer #6
    conv6 = tf.layers.conv2d(
    	inputs=conv5,
    	filters=256,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    #Pooling Layer #3
    pool3 = tf.layers.average_pooling2d(inputs=conv6, pool_size=[2,2], strides=2, padding='same')
    
    tf.summary.image("Conv6_Layer", tf.transpose(tf.expand_dims(pool3[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    #Convolutional Layer #7
    conv7 = tf.layers.conv2d(
    	inputs=pool3,
    	filters=384,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("Conv7_Layer", tf.transpose(tf.expand_dims(conv7[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    #Convolutional Layer #8
    conv8 = tf.layers.conv2d(
    	inputs=conv7,
    	filters=512,
    	kernel_size=[3,3],
    	padding="valid",
    	activation=tf.nn.relu)
    	
    tf.summary.image("Conv8_Layer", tf.transpose(tf.expand_dims(conv8[0], 0), [3, 1, 2, 0]), max_outputs=5)
    
    #print(conv8.shape)
    
    #Dense Layer
    conv8_flat = tf.contrib.layers.flatten(conv8)
    
    #Add additional feature to dense layers (speed)
    conv8_flat = tf.concat([tf.reshape(speed, [-1, 1]), conv8_flat], axis=1)
    
    print(conv8_flat.shape)
    
    dense1 = tf.layers.dense(inputs=conv8_flat, units=4096, activation=tf.nn.relu)
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
    
    
    
