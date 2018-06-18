import pygame
import os
import sys
import time

pygame.display.init()
pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
joysticks[0].init()

numaxes=joysticks[0].get_numaxes()

while 1:
	pygame.event.pump()
	os.system('cls')
	print(joysticks[0].get_name())
	for i in range(numaxes):
		print("Axis %d: %.2f" % (i, joysticks[0].get_axis(i)))
	time.sleep(0.1)