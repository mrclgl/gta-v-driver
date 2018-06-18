import csv
import os
import sys

if not (len(sys.argv) > 1):
	print('\nNo csv data file folder path given!\n=>convert_relpaths_to_abspaths.py <folder-path>')
	sys.exit(0)
else:
	folder_path=sys.argv[1] + '/'

new_csv_file=open(folder_path + 'abspaths_data.csv', "w+")
	
with open(folder_path + 'data.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
      label=row[0]
      speed=row[1]
      path=row[2]
      path=os.path.abspath(path)
      new_csv_file.write("%s,%s,%s\n" % (label, speed, path))
	  
new_csv_file.close()

os.rename(folder_path + 'data.csv', folder_path + 'relpaths_data.csv')
os.rename(folder_path + 'abspaths_data.csv', folder_path + 'data.csv')