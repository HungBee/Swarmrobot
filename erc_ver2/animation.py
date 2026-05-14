import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Circle
import numpy as np
from config import *

with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file)

robot_data = data[:NUM_ROBOT]
membrane_history = None
if len(data) > NUM_ROBOT and 'membrane_history' in data[-1]:
    membrane_history = data[-1]['membrane_history']

max_frames = robot_data[0]['path'].shape[0]

plt.figure(figsize=(10, 5))
ax = plt.axes()

for frame in range(0, max_frames, 2):
    ax.cla()

    # Vẽ vật cản
    for obs in OBSTACLES:
        ax.add_patch(MplPolygon(obs, color='grey'))

    # Vẽ màng nếu có dữ liệu lịch sử
    if membrane_history is not None:
        idx = min(frame, len(membrane_history) - 1)
        nodes = membrane_history[idx]
        ax.add_patch(MplPolygon(nodes, fill=False, edgecolor='magenta', linewidth=2))

    # Vẽ robot và đường đi
    for i in range(NUM_ROBOT):
        path = robot_data[i]['path']
        ax.plot(path[:frame, 1], path[:frame, 2], label=f"Robot {i}")
        pose = path[frame, :]
        circle = Circle(pose[1:3], ROBOT_RADIUS, fill=False, color='blue')
        ax.add_patch(circle)
        ax.arrow(pose[1], pose[2], pose[4] * 0.3, pose[5] * 0.3, width=0.02, color='k')

    ax.axis('scaled')
    ax.grid(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.legend(loc='upper right')
    center = np.mean([robot_data[i]['path'][frame, 1:3] for i in range(NUM_ROBOT)], axis=0)
    plt.xlim(center[0] - 5, center[0] + 5)
    plt.ylim(center[1] - 3, center[1] + 3)
    plt.tight_layout()
    plt.pause(0.001)

plt.show()

