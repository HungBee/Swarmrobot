import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from matplotlib.patches import Polygon
from config import *

shape = 1
methods = ['erc', 'bc', 'iapf']

for method in methods:
    file_name = "results/data_{}_shape{}.txt".format(method, shape)
    with open(file_name, 'rb') as file:
        data = pickle.load(file)

    # Mean vel and energy consumption
    acc = 0; vel = 0
    for i in range(NUM_ROBOT):
        path = data[i]['path']
        total_time = path[-1,0]
        velocity = path[:,4:7]
        acceleration = path[:,7:10]
        vel += np.linalg.norm(velocity)/total_time
        acc += np.linalg.norm(acceleration)
    
    # Heading
    headings = []
    for i in range(1,len(path)):
        heading = 0
        for j in range(NUM_ROBOT):
            heading += data[j]['path'][i,4:6]/np.linalg.norm(data[j]['path'][i,4:6])
        headings.append(np.linalg.norm(heading)/NUM_ROBOT)
    
    print(method, " - time: ", len(path)*TIMESTEP)
    print(method, " - order: ", np.mean(headings))
    print(method, " - speed: ", vel/NUM_ROBOT)
    print(method, " - acceleration: ", acc**2/NUM_ROBOT)