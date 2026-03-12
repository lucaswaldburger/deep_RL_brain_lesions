import logging

from lib.DQL_visualization_actions import visualizing_seq_act

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
    xmin = float(row[1])
    ymin = float(row[2])
    xmax = float(row[3])
    ymax = float(row[4])

    visualizing_seq_act(
        'default_model',
        '../' + dir_path + '/T1ce.jpeg_3d.jpeg',
        [xmin, ymin, xmax, ymax],
        dir_path.split('/')[-1],
    )
    counter += 1
    if counter >= 50:
        break
