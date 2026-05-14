import numpy as np
import pickle
from config import *
from utils import *
from robot_erc import Robot
from membrane import ElasticMembrane

def collision(robots):
    for i in range(NUM_ROBOT):
        pos = robots[i].position
        for obs in OBSTACLES:
            near = nearest_point_to_obstacle(pos[:2], obs)
            if np.linalg.norm(pos[:2]-near) < ROBOT_RADIUS:
                print("Obstacle collision")
                return True
    for i in range(NUM_ROBOT-1):
        for j in range(i+1, NUM_ROBOT):
            if np.linalg.norm(robots[i].position - robots[j].position) < 2*ROBOT_RADIUS:
                print("Inter-agent collision")
                return True
    return False

if __name__ == "__main__":
    robots = [Robot(i, INITS[i,:]) for i in range(NUM_ROBOT)]

    init_center = np.mean(INITS[:, :2], axis=0)
    init_radius = max(np.linalg.norm(INITS[:,:2] - init_center, axis=1)) + ROBOT_RADIUS + 0.5
    membrane = ElasticMembrane(init_center, init_radius)

    membrane_history = [membrane.get_nodes().copy()]

    for iter in range(ITER_MAX):
        if iter % 50 == 0:
            print(f"Iteration {iter}")

        membrane.update(OBSTACLES, robots, dt=TIMESTEP)
        membrane_history.append(membrane.get_nodes().copy())

        for r in robots:
            r.compute_control(robots, dt=TIMESTEP, membrane=membrane)

        if collision(robots):
            print("Collision! Dừng.")
            break

        if all(r.position[0] > XGOAL for r in robots):
            print(f"Đã đến đích sau {iter} vòng lặp!")
            break

    with open(FILE_NAME, 'wb') as f:
        data = [{'path': np.array(r.path)} for r in robots]
        data.append({'membrane_history': membrane_history})
        pickle.dump(data, f)
