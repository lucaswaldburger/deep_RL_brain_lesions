import sys

if "./lib" not in sys.path:
    sys.path.append("./lib")

from lib.DQL_visualization_layers import *
f = open('../outputs.txt','r')
data = f.read()
data = data[:-1]
data = [col.split(',') for col in data.split('\n')]

counter = 0
for row in data:
    dir = row[0]
    for layer in range(1,4):
        visualize_layers('default_model',
        '../'+dir+'/T1ce.jpeg_3d.jpeg',
        str(layer))
    counter += 1
    if counter >= 50:
        break