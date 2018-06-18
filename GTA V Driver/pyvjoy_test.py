import pyvjoy
import time
import math

j = pyvjoy.VJoyDevice(1)

vjoy_max = 32768

while 1:
	j.data.wAxisX = int(0 * vjoy_max)
	j.data.wAxisY = int(0 * vjoy_max)
	j.data.wAxisZ = int(0 * vjoy_max)

	j.update()