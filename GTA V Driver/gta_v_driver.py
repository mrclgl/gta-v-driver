from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from gta_v_driver_model import gta_v_driver_model
from gta_v_driver_model import learning_rate

batch_size=64
eval_batch_size=64

with_layer_images=False

def cnn_model_fn(features, labels, mode):

	if mode != learn.ModeKeys.INFER:
		current_batch_size = batch_size
	else:
		current_batch_size = 1

	predictions = gta_v_driver_model(tf.reshape(features, [current_batch_size, -1]), mode)
	
	loss = None
	train_op = None
	global_step = tf.contrib.framework.get_global_step()
	
	#Calculate Loss (for both TRAIN and EVAL modes)
	if mode != learn.ModeKeys.INFER:
		loss = tf.losses.mean_squared_error(labels=tf.reshape(labels, [-1, 3]), predictions=predictions)
	
	#Configure the Training Op (for TRAIN mode)
	if mode == learn.ModeKeys.TRAIN:
		train_op = tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=global_step,
			learning_rate=learning_rate,
			optimizer="Adam")
			
		#train_op = tf.Print(train_op, [loss], message = 'Loss: ')
			
			
	#Generate Predictions
	predictions = {
		"predictions": predictions
	}
	
	#Return a ModelFnOps object
	return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)