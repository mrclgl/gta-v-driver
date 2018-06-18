import sys
import random
import csv
from collections import Counter
import numpy as np
import os

if not (len(sys.argv) > 1):
	print('\nNo data paths given!\n=>data_selector.py <output-data-path> <input-data-paths>')
	sys.exit(0)
else:
	output_data_path=sys.argv[1]
	data_paths=sys.argv[2:]

if os.path.exists(output_data_path + '/data.csv'):
	print('Output data file already exists!')
	answer = input('Do you want to override it? (Y/N): ')
	if (answer == 'N' or answer == 'n'):
		sys.exit(0)
	
output_file = open(output_data_path, 'w+')

for i in range(len(data_paths)):
	
	print("\n%s:" % data_paths[i])
	
	entries=[]
	right_samples=0
	left_samples=0
	straight_samples=0
	
	with open(data_paths[i]) as csvfile:
		csv_reader = csv.reader(csvfile, delimiter=',')
		for row in csv_reader:
			steering_angle=float(row[0])
			
			if (steering_angle > 0.5):
				right_samples+=1
			elif (steering_angle < 0.5):
				left_samples+=1
			else:
				straight_samples+=1
				
			throttle=float(row[1])
			brake=float(row[2])
			entries.append((steering_angle, throttle, brake))
	
	counter = Counter(tuple(tup) for tup in entries)
	
	most_common=counter.most_common(20)
	
	print("Total Samples: %d\n" % len(entries))
	print("Average counts: %.3f" % np.mean(list(counter.values())))
	print("Average counts (most common): %.3f\n" % np.mean([count for key, count in most_common]))
	
	print("Right steer samples: %d (%.3f%% of total samples)" % (right_samples, (right_samples/len(entries))*100))
	print("Left steer samples: %d (%.3f%% of total samples)" % (left_samples, (left_samples/len(entries))*100))
	print("Straight steer samples: %d (%.3f%% of total samples)\n" % (straight_samples, (straight_samples/len(entries))*100))
	
	for index in range(len(most_common)):
		elem = most_common[index]
		print("%d.\t%.3f%%  [%.6f  %.6f  %.6f] -> %d" % (index+1, 100*(elem[1]/len(entries)), elem[0][0], elem[0][1], elem[0][2], elem[1]))
	
	with open(data_paths[i], 'r') as file:
		for line in file:
			output_file.write(line)
	