import sys

if "./lib" not in sys.path:
    sys.path.append("./lib")

from lib.DQL_visualization_actions import *
f = open('../outputs.txt','r')
data = f.read()
data = data[:-1]
data = [col.split(',') for col in data.split('\n')]

counter = 0
for row in data:
    dir = row[0]
    xmin = float(row[1])
    ymin = float(row[2])
    xmax = float(row[3])
    ymax = float(row[4])

    visualizing_seq_act('default_model',
    '../'+dir+'/T1ce.jpeg_3d.jpeg',
	[xmin, ymin, xmax, ymax],
	dir.split('/')[-1])
    counter += 1
    if counter >= 50:
        break