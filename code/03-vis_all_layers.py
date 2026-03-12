import logging

from lib.DQL_visualization_layers import visualize_layers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

with open('../outputs.txt', 'r') as f:
    data = f.read().rstrip()

rows = [col.split(',') for col in data.split('\n')]

counter = 0
for row in rows:
    dir_path = row[0]
    for layer in range(1, 4):
        visualize_layers(
            'default_model',
            '../' + dir_path + '/T1ce.jpeg_3d.jpeg',
            str(layer),
        )
    counter += 1
    if counter >= 50:
        break
