import numpy as np
from config import *
from utils import nearest_point_on_polygon, point_inside_polygon, nearest_point_to_obstacle

class Robot():
    def __init__(self, index, position, velocity=np.zeros(3)):
        self.index = index
        self.stamp = 0.0
        self.position = position
        self.velocity = velocity
        self.control = np.zeros(3)
        self.path = [np.concatenate([[self.stamp], self.position, self.velocity, self.control])]

    def update_state(self, control, dt):
        control_norm = np.linalg.norm(control)
        if control_norm > UMAX:
            control = control / control_norm * UMAX
        velocity = self.velocity + control * dt
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > VMAX:
            velocity = velocity / velocity_norm * VMAX
        position = self.position + velocity * dt
        self.stamp += dt
        self.position = position
        self.velocity = velocity
        self.control = control
        self.path.append(np.concatenate([[self.stamp], self.position, self.velocity, self.control]))

    def compute_control(self, robots, membrane, dt):
        v_mig = self.behavior_migration()
        v_form = self.behavior_formation(robots)
        v_obs = self.behavior_obstacle()
        v_col = self.behavior_collision(robots)
        v_memb = self.behavior_membrane(membrane)
        desired_velocity = v_mig + v_form + v_obs + v_col + v_memb
        desired_control = (desired_velocity - self.velocity) / dt
        self.update_state(desired_control, dt)

    def behavior_migration(self):
        return VREF * UREF

    def behavior_formation(self, robots):
        v_form = np.zeros(3)
        for i in range(NUM_ROBOT):
            v_form += (robots[i].position - self.position) - (TOPOLOGY[i, :] - TOPOLOGY[self.index, :])
        return W_form * v_form

    def behavior_obstacle(self):
        v_obs = np.zeros(3)
        for obstacle in OBSTACLES:
            obs_point = nearest_point_to_obstacle(self.position[:2], obstacle)
            obs_rel = self.position - np.concatenate([obs_point, [self.position[2]]])
            obs_dis = np.linalg.norm(obs_rel)
            if obs_dis < ALERT_RADIUS:
                v_obs += 0.5 * (1 / obs_dis - 1 / ALERT_RADIUS) / (obs_dis ** 2) * obs_rel / obs_dis
        return W_obs * v_obs

    def behavior_collision(self, robots):
        v_col = np.zeros(3)
        for i in range(NUM_ROBOT):
            if i == self.index:
                continue
            pos_rel = self.position - robots[i].position
            pos_dis = np.linalg.norm(pos_rel)
            if pos_dis < ALERT_RADIUS:
                v_col += 2 * (1 / pos_dis - 1 / ALERT_RADIUS) / (pos_dis ** 2) * pos_rel / pos_dis
        return W_col * v_col

    def behavior_membrane(self, membrane):
        v_memb = np.zeros(3)
        nodes = membrane.get_nodes()
        pos2d = self.position[:2]
        inside = point_inside_polygon(pos2d, nodes)
        if not inside:
            nearest = nearest_point_on_polygon(pos2d, nodes)
            vec = nearest - pos2d
            dist = np.linalg.norm(vec)
            if dist > 0:
                v_memb[:2] = W_membrane * (vec / dist) * min(dist, 1.5)
        return v_memb
