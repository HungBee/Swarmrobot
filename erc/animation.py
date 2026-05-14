import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle
import numpy as np
from config import *

def get_circle(x, y, r):
    theta = np.linspace(0, 2 * np.pi, 150)
    a = x + r * np.cos(theta)
    b = y + r * np.sin(theta)
    return a, b

percent = 0.3
width = 0.02
export = True   # False để xem trực tiếp
if export:
    import cv2
    image_array = []

with open(FILE_NAME, 'rb') as file:
    data = pickle.load(file)

robot_data = data[:NUM_ROBOT]
region_history = data[-1]['region_history']

max_frames = robot_data[0]['path'].shape[0]

plt.figure(figsize=(10, 5))
ax = plt.axes()
for frame in range(0, max_frames, 2):
    ax.cla()

    # Vẽ vật cản
    for obs in OBSTACLES:
        ax.add_patch(Polygon(obs, color='grey'))

    # Vẽ vùng ảo (đồng bộ frame)
    # Nếu frame vượt quá số region đã lưu, lấy region cuối cùng
    idx = min(frame, len(region_history)-1)
    reg = region_history[idx]
    if reg['mode'] == 'circle':
        circ = Circle(reg['center'], reg['a'], fill=False, edgecolor='magenta', linewidth=2)
        ax.add_patch(circ)
    else:
        ell = Ellipse(xy=reg['center'], width=2*reg['a'], height=2*reg['b'],
                      angle=np.rad2deg(reg['theta']), fill=False, edgecolor='magenta', linewidth=2)
        ax.add_patch(ell)

    # Vẽ robot và đường đi
    for i in range(NUM_ROBOT):
        path = robot_data[i]['path']
        ax.plot(path[:frame, 1], path[:frame, 2], label=f"Robot {i}")
        pose = path[frame, :]
        a_circ, b_circ = get_circle(pose[1], pose[2], ROBOT_RADIUS)
        ax.plot(a_circ, b_circ, '-k')
        plt.arrow(pose[1], pose[2], pose[4]*percent, pose[5]*percent, width=width, color='k')

    ax.axis('scaled')
    ax.grid(True)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.legend(loc='upper right')

    center = np.mean([robot_data[i]['path'][frame, 1:3] for i in range(NUM_ROBOT)], axis=0)
    plt.xlim(center[0]-5, center[0]+5)
    plt.ylim(center[1]-3, center[1]+3)
    plt.tight_layout()

    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])
    plt.pause(0.001)

    if export:
        plt.savefig("results/data.png")
        img = cv2.imread("results/data.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_array.append(img)

if export:
    import imageio
    imageio.mimsave(GIF_NAME, image_array)

plt.show()
