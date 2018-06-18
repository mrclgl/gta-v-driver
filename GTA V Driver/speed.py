import time
import os

while True:
	file = open("speed.txt", "r")
	os.system('cls')
	print(file.read())
	file.close()
	time.sleep(0.02)