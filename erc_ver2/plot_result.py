import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from config import *

def get_circle(x,y,r):
    theta = np.linspace(0, 2 * np.pi, 20)
    a = x + r * np.cos( theta )
    b = y + r * np.sin( theta )
    return a, b

shape = 2
controller = 'erc'
file_name = "results/data_{}_shape{}.txt".format(controller, shape)
with open(file_name, 'rb') as file:
    data = pickle.load(file)

def plot_path():
    fig = plt.figure(figsize=(10,3))
    ax = fig.subplots()

    # Plot obstacles
    for i in range(len(OBSTACLES)):
        ax.add_patch(Polygon(OBSTACLES[i],color='dimgrey'))

    for i in range(NUM_ROBOT):
        path = data[i]['path']
        velocity = path[:,4:6]
        speed = np.linalg.norm(velocity, axis=1)
        pp = ax.scatter(path[:,1], path[:,2], c=speed, cmap=cmap.summer, marker='.', s=20)

    fig.colorbar(pp, ax=ax, orientation="horizontal", fraction=0.1, label="Speed (m/s)")

    # Plot formation
    n_plot = 5
    length = path.shape[0]
    for i in range(n_plot+1):
        iter = int(round(length/n_plot)*i)
        if iter >= length:
            iter = length-1

        # Plot robots
        for i in range(NUM_ROBOT):
            a, b = get_circle(data[i]['path'][iter,1], data[i]['path'][iter,2], ROBOT_RADIUS)
            plt.plot(a, b, '-b')

        positions = []
        for i in range(NUM_ROBOT):
            positions.append(data[i]['path'][iter,1:4])

        # Plot current formation
        positions.append(positions[0])
        formation = np.array(positions)
        plt.plot(formation[:,0], formation[:,1], '-k')

    plt.axis('scaled')
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.tight_layout()
    plt.savefig("results/path_{}_shape{}.pdf".format(controller, shape), format="pdf", bbox_inches="tight")

def plot_mode():
    fig = plt.figure(figsize=(6,2.5))
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)
    size = 10
    path = data[0]['path']
    x = np.arange(0, path.shape[0], size)
    num_state = 0
    for i in range(NUM_ROBOT):
        num_state += data[i]['path'][:,10]
    num_state = num_state[x]

    scales = []
    for i in range(NUM_ROBOT):
        scale = data[i]['path'][:,11]
        scale[scale==-1]=0
        scales.append(scale)
    scales = np.array(scales).T

    ax.bar(path[:,0][x], num_state, label="Tailgating")
    ax.bar(path[:,0][x], NUM_ROBOT-num_state, bottom=num_state, label="Formation")
    ax2.fill_between(path[:,0][x], np.min(scales,axis=1)[x], np.max(scales,axis=1)[x], color="k", label="Max/Min", alpha=0.3)
    ax2.plot(path[:,0][x], np.mean(scales,axis=1)[x], 'k-', label="Average")
    ax2.grid(True, axis='y')
    ax.set_xlabel("Time (s)")
    ax2.set_ylabel("Scaling factor $\kappa$")
    ax.set_ylabel("Number of Robots")
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right') 
    ax2.tick_params(bottom=False, labelbottom=False)
    plt.xlim([0, path[-1,0]])
    ax.set_ylim([0, NUM_ROBOT])
    ax2.set_ylim(-0.1, 1.1)
    plt.tight_layout()
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.32))
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.32))
    plt.savefig("results/mode_{}_shape{}.pdf".format(controller, shape), format="pdf", bbox_inches="tight")

def plot_order():
    plt.figure(figsize=(6,2.5))
    path = data[0]['path']
    headings = []
    for i in range(1,len(path)):
        heading = 0
        for j in range(NUM_ROBOT):
            heading += data[j]['path'][i,4:6]/np.linalg.norm(data[j]['path'][i,4:6])
        headings.append(np.linalg.norm(heading)/NUM_ROBOT)

    headings = np.array(headings)
    thresh = 0.9998
    x = np.where(headings>thresh)[0]
    
    plt.plot(path[1:,0], headings, 'r-')
    plt.scatter(path[x,0], headings[x], c='b', marker='.', s=5, zorder=10)
    plt.xlabel("Time (s)")
    plt.ylabel("Order")
    plt.xlim([0, path[-1,0]])
    plt.ylim(0,1.1)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("results/order_{}_shape{}.pdf".format(controller, shape), format="pdf", bbox_inches="tight")

plot_path()
plot_order()

if controller == "erc":
    plot_mode()
plt.show()