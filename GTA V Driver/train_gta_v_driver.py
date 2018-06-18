from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import cv2

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from gta_v_driver_model import width
from gta_v_driver_model import height

from gta_v_driver import cnn_model_fn
from gta_v_driver_multigpu import cnn_model_fn_multigpu

tf.app.flags.DEFINE_bool('multi_gpu', False, 'Use multi-gpu version for training?')

tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs should we use?')

flags = tf.app.flags.FLAGS

if flags.multi_gpu:
    from gta_v_driver_multigpu import batch_size
    from gta_v_driver_multigpu import eval_batch_size
else:
    from gta_v_driver import batch_size
    from gta_v_driver import eval_batch_size
	
def main(argv):

	if len(argv) == 2:
		if (argv[1] == "eval"):
			eval_model()
		else:
			train_model(int(argv[1]))
	elif len(argv) > 3:
		if (argv[1] == "predict"):
			predict(argv[2], float(argv[3]))
		elif (argv[1] == "visualize"):
			visualize_model(argv[2], float(argv[3]))
	else:
		print("... <number-of-steps>")
		print("... predict <image-path>")
		print("... eval")

def get_model_fn():
	if flags.multi_gpu:
		model_fn = cnn_model_fn_multigpu
	else:
		model_fn = cnn_model_fn
	
	return model_fn
				
def predict(image_name, speed):
	#Create the Estimator
	gta_driver = learn.Estimator(model_fn=get_model_fn(), model_dir="/tmp/gta_driver_model")

	filename_queue = tf.train.string_input_producer([image_name])

	reader = tf.WholeFileReader()
	key, value = reader.read(filename_queue)
	
	image = tf.image.decode_bmp(value, channels=3)
	
	init_op = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init_op)
		
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		image = tf.reshape(image, [width * height * 3])
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		image = image.eval()
		
		speed = tf.reshape(speed, [1]).eval()
		
		features = tf.concat([speed, image], axis=0).eval()
	
		coord.request_stop()
		coord.join(threads)
	
	predictions = gta_driver.predict(x=features, as_iterable=True)
	
	for i, p in enumerate(predictions):
		print("Predictions [Steering Angle: %s, Throttle: %s, Brake: %s]" % (p["predictions"][0], p["predictions"][1], p["predictions"][2]))
	
def train_model(steps):
    tf.logging.set_verbosity(tf.logging.INFO)
    
    config = tf.contrib.learn.RunConfig(save_summary_steps=25, save_checkpoints_secs=180)

    #Create the Estimator
    gta_driver = learn.Estimator(model_fn=get_model_fn(), model_dir="/tmp/gta_driver_model", config=config)
	
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    	input_fn=eval_input_fn,
    	eval_steps=1,
    	every_n_steps=10
    	)
	
    #Train the model
    gta_driver.fit(
    	input_fn=train_input_fn,
    	steps=steps,
		monitors=[validation_monitor]
		)
    
    eval_model()

def eval_model():
	tf.logging.set_verbosity(tf.logging.INFO)

	#Create the Estimator
	gta_driver = learn.Estimator(model_fn=get_model_fn(), model_dir="/tmp/gta_driver_model")
	
	#Evaluate the model and print results
	eval_results = gta_driver.evaluate(
		input_fn=eval_input_fn,
		steps=10
		)
		
	print(eval_results)
	
# Function to tell TensorFlow how to read a single image from input file
def read_data_example(filename_queue):
	
    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example 
    key, fullExample = recordReader.read(filename_queue)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
			'image/speed': tf.VarLenFeature(tf.float32),
            'image/label': tf.VarLenFeature(tf.float32),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })

    image_buffer = features['image/encoded']
    speed = tf.reshape(tf.sparse_tensor_to_dense(features['image/speed']), [1])
    label = tf.reshape(tf.sparse_tensor_to_dense(features['image/label']), [3])

    # Decode the jpeg
    with tf.name_scope('decode_bmp',[image_buffer], None):
        # decode
        image = tf.image.decode_bmp(image_buffer, channels=3)
    
        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		
        image = tf.image.per_image_standardization(image)


    # cast image into a single array
    image = tf.reshape(image,[height*width*3])
	
    features = tf.concat([speed, image], axis=0)
    
    return features, label
    
def gen_data_filenames(name, num_shards):
	return list('data/%s-%.5d-of-%.5d' % (name, shard, num_shards) for shard in range(num_shards))
	
def train_input_fn():
	# convert filenames to a queue for an input pipeline.
	filename_queue = tf.train.string_input_producer(gen_data_filenames('train', 580), num_epochs=None)
	
	# associate the "label" and "image" objects with the corresponding features read from 
	# a single example in the training data file
	features, label = read_data_example(filename_queue)
	
	# associate the "label_batch" and "image_batch" objects with a randomly selected batch---
	# of labels and images respectively
	featuresBatch, labelBatch = tf.train.shuffle_batch(
		[features, label],
		batch_size=batch_size,
		num_threads=40,
		capacity=4000,
		min_after_dequeue=500)
	
	return featuresBatch, labelBatch
	
def eval_input_fn():
	# convert filenames to a queue for an input pipeline.
	filename_queue = tf.train.string_input_producer(gen_data_filenames('validation', 15), num_epochs=None)

	# associate the "label" and "image" objects with the corresponding features read from 
	# a single example in the validation data file
	vfeatures, vlabel = read_data_example(filename_queue)
	
	# associate the "vimageBatch" and "vlabelBatch" objects with a randomly selected batch---
	# of labels and images respectively
	vfeaturesBatch, vlabelBatch = tf.train.shuffle_batch(
		[vfeatures, vlabel],
		batch_size=eval_batch_size,
		num_threads=10,
		capacity=1000,
		min_after_dequeue=300)
	
	return vfeaturesBatch, vlabelBatch
	
if __name__ == "__main__":
	tf.app.run()
