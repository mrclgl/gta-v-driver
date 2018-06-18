import csv
import os
import sys

if not (len(sys.argv) > 1):
	print('\nNo csv data file folder path given!\n=>convert_abspaths_to_relpaths.py <folder-path>')
	sys.exit(0)
else:
	folder_path=sys.argv[1] + '/'

new_csv_file=open(folder_path + 'relpaths_data.csv', "w+")

with open(folder_path + 'data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      steering_angle=row[0]
      throttle=row[1]
      brake=row[2]
      speed=row[3]
      path=row[4]
      path=os.path.relpath(path)
      new_csv_file.write("%s,%s,%s,%s,%s\n" % (steering_angle, throttle, brake, speed, path))
	  
new_csv_file.close()

os.rename(folder_path + 'data.csv', folder_path + 'abspaths_data.csv')
os.rename(folder_path + 'relpaths_data.csv', folder_path + 'data.csv')