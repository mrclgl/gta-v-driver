import time
import csv
import win32gui, win32api
import os
import sys
import PIL
from PIL import Image
import mss
import pygame

if not (len(sys.argv) > 2):
	print('\nNo save path given!\n=>data_recorder.py <save-path> <speed.txt-path>')
	sys.exit(0)
else:
	save_path=sys.argv[1] + '/'
	speed_fil_path=sys.argv[2]

max_samples=100000
samples_per_second=10

if not os.path.exists(save_path):
    os.makedirs(save_path)

csv_file = open(save_path + 'data.csv', 'w+')

print('Recording starts in 5 seconds...')
time.sleep(5)
print('Recording started!')

current_sample=0
last_time=0
start_time=time.time()
wait_time=(1/samples_per_second)
stats_frame=0

w = 0
a = 0
s = 0
d = 0
wa = 0
wd = 0
sa = 0
sd = 0
nk = 0

sct = mss.mss()
mon = {'top': 0, 'left': 0, 'width': 800, 'height': 600}

pause=False
return_was_down=False
speed='0.0'

pygame.display.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
joysticks[0].init()

while True:
	pygame.event.pump()
	
	if (win32api.GetAsyncKeyState(0x24)&0x8001 > 0):
			break

	if (win32api.GetAsyncKeyState(0x0D)&0x8001 > 0):
		if (return_was_down == False):
			if (pause == False):
				pause = True
			else:
				pause = False
				
		return_was_down = True
	else:
		return_was_down = False

	if (time.time() - last_time >= wait_time):
	
		fps=1 / (time.time() - last_time)
		last_time = time.time()
		
		stats_frame+=1
		if (stats_frame >= 10):
			stats_frame=0
			os.system('cls')
			print('FPS: %.2f Total Samples: %d Time: %s' % (fps, current_sample, time.strftime("%H:%M:%S",time.gmtime(time.time() - start_time))))
			if (pause == False):
				print('Status: Recording')
			else:
				print('Status: Paused')
		
		file = open(speed_fil_path, "r")
		new_speed = file.read()
		file.close()
		
		if (len(new_speed) > 0):
			speed = new_speed
		
		if (pause):
			time.sleep(0.01)
			continue
		
		sct_img = sct.grab(mon)
		img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
		img = img.resize((640, 360), PIL.Image.BICUBIC)
		img = img.crop(box=(0, 200, 640, 360))
		
		steering_angle=joysticks[0].get_axis(0)
		if (abs(steering_angle) < 0.008):
			steering_angle=0
		steering_angle=(steering_angle+1)/2
		if (steering_angle > 0.99):
			steering_angle=1
		if (steering_angle < 0.01):
			steering_angle=0
			
		throttle=joysticks[0].get_axis(2)
		if (throttle > 0.98):
			throttle=1
		throttle=1-(throttle+1)/2
			
		brake=joysticks[0].get_axis(3)
		if (brake > 0.98):
			brake=1
		brake=1-(brake+1)/2
					
		path = save_path + 'img%d.bmp' % current_sample
		img.save(path, 'BMP')
		csv_file.write('%f,%f,%f,%s,%s\n' % (steering_angle, throttle, brake, speed, path))
			
		current_sample += 1
		
		if (current_sample >= max_samples):
			break
	
	
print('\nDONE')
print('Total Samples: %d\n' % current_sample)

joysticks[0].quit()
pygame.display.quit()
pygame.joystick.quit()