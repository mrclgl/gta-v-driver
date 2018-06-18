from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
import mss
import win32gui, win32api
import time
import PIL
import pyvjoy
import cv2

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from gta_v_driver import cnn_model_fn
from gta_v_driver_multigpu import cnn_model_fn_multigpu
from directkeys import PressKey
from directkeys import ReleaseKey
from PIL import Image

tf.app.flags.DEFINE_bool('multi_gpu', False, 'Use multi-gpu version for predicting?')

tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs should we use?')

flags = tf.app.flags.FLAGS

j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768

class FPSTimer:
	def __init__(self):
		self.t = time.time()
		self.iter = 0
		
	def reset(self):
		self.t = time.time()
		self.iter = 0
		
	def on_frame(self):
		self.iter += 1
		if self.iter == 100:
			e = time.time()
			print('FPS: %0.2f' % (100.0 / (e - self.t)))
			self.t = time.time()
			self.iter = 0

class FastPredict:
    
    def _createGenerator(self):
        while True:
            yield self.next_features

    def __init__(self, estimator):
        self.estimator = estimator
        self.first_run = True
        
    def predict(self, features):
        self.next_features = features
        if self.first_run:
            self.predictions = self.estimator.predict(x = self._createGenerator())
            self.first_run = False
        return next(self.predictions)

def lerp(a, b, t):
	return (t * a) + ((1-t) * b)
		
def predict_loop():

	timer = FPSTimer()
	
	pause=True
	return_was_down=False
	
	if flags.multi_gpu:
		model_fn = cnn_model_fn_multigpu
	else:
		model_fn = cnn_model_fn
	
	config = learn.RunConfig(gpu_memory_fraction=0.4)
	
	gta_driver = FastPredict(learn.Estimator(model_fn=model_fn, model_dir="/tmp/gta_driver_model", config=config))

	sct = mss.mss()
	mon = {'top': 0, 'left': 0, 'width': 800, 'height': 600}
	
	speed=0
	
	print('Ready')
	
	while True:
		
		if (win32api.GetAsyncKeyState(0x08)&0x8001 > 0):
			break
		
		if (win32api.GetAsyncKeyState(0x0D)&0x8001 > 0):
			if (return_was_down == False):
				if (pause == False):
					pause = True
					
					j.data.wAxisX = int(vjoy_max * 0.5)
					j.data.wAxisY = int(vjoy_max * 0)
					j.data.wAxisZ = int(vjoy_max * 0)

					j.update()
					
					print('Paused')
				else:
					pause = False
					
					print('Resumed')
				
			return_was_down = True
		else:
			return_was_down = False
		
		if (pause):
			time.sleep(0.01)
			continue
		
		sct_img = sct.grab(mon)
		img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
		img = img.resize((640, 360), PIL.Image.BICUBIC)
		img = img.crop(box=(0, 200, 640, 360))
		img = np.array(img)
		img = img.astype(np.float32, copy=False)
		img = np.divide(img, 255)
		
		img = np.reshape(img, (640*160*3))
		
		img = (img - np.mean(img)) / max(np.std(img), 1.0/np.sqrt(img.size))
		
		try:
			file = open("speed.txt", "r")
			speed = float(file.read())
			file.close()
		except (ValueError):
			pass
		
		features = np.concatenate((np.array([speed], dtype=np.float32), img))
		
		predictions = gta_driver.predict(features)
		
		j.data.wAxisX = int(vjoy_max * min(max(predictions["predictions"][0], 0), 1))
		j.data.wAxisY = int(vjoy_max * min(max(predictions["predictions"][1], 0), 1))
		j.data.wAxisZ = int(vjoy_max * min(max(predictions["predictions"][2], 0), 1))
		
		j.update()
		
		os.system('cls')
		print("Steering Angle: %.2f" % min(max(predictions["predictions"][0], 0), 1))
		print("Throttle: %.2f" % min(max(predictions["predictions"][1], 0), 1))
		print("Brake: %.2f" % min(max(predictions["predictions"][2], 0), 1))
		
		#timer.on_frame()

if __name__ == "__main__":
	predict_loop()
