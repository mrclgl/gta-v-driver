from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from gta_v_driver_model import gta_v_driver_model
from gta_v_driver_model import learning_rate

batch_size=128
eval_batch_size=128

def cnn_model_fn_multigpu(features, labels, mode):
	
	num_gpus=tf.app.flags.FLAGS.num_gpus
	
	reuse=False
	train_op=None
	
	for i in range(num_gpus):
		with tf.device('/gpu:%d' % i):
			with tf.variable_scope('ConvNet', reuse=reuse):
			
				tower_grads = []
				tower_predictions = []
	
				if mode != learn.ModeKeys.INFER:
					tower_loss = []
				else:
					tower_loss = None
			
				input_layer = tf.split(features, num_gpus)
				
				if mode != learn.ModeKeys.INFER:
					gpu_batch_size = int(batch_size/num_gpus)
				else:
					gpu_batch_size = 1
				
				if mode != learn.ModeKeys.INFER:
					labels = tf.split(labels, num_gpus)
				
				#Model Function
				predictions = gta_v_driver_model(tf.reshape(input_layer[i], [gpu_batch_size, -1]), mode)
				
				tower_predictions.append(predictions)
				
				if mode != learn.ModeKeys.INFER:
					loss = tf.losses.mean_squared_error(labels=tf.reshape(labels[i], [gpu_batch_size, 3]), predictions=predictions)
					tower_loss.append(loss)
				
				if mode == learn.ModeKeys.TRAIN:
					optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
					grads = optimizer.compute_gradients(loss)
				
					tower_grads.append(grads)
				
				reuse=True
			
	with tf.device('/cpu:0'):
		if mode == learn.ModeKeys.TRAIN:
			tower_grads = average_gradients(tower_grads)
			global_step = tf.contrib.framework.get_global_step()
			train_op = optimizer.apply_gradients(tower_grads, global_step=global_step)
		
		if mode != learn.ModeKeys.INFER:
			tower_loss = tf.reduce_sum(tower_loss)/num_gpus
			tf.summary.scalar('Loss', tower_loss)
			tower_loss = tf.Print(tower_loss, [tower_loss])
			
		tower_predictions = tf.reshape(tower_predictions, [-1, 3])

		#Generate Predictions
		predictions = {
			"predictions": tower_predictions
		}
	
		#Return a ModelFnOps object
		return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=tower_loss, train_op=train_op)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
