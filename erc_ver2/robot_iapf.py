import numpy as np
import math

from config import *
from utils import *
from enum import Enum

class Mode(Enum):
    FORMATION = 0
    TAILGATING = 1

class Robot():
    def __init__(self, index, position, velocity=np.zeros(3)):
        self.index = index
        
        self.stamp = 0.0
        self.position = position
        self.velocity = velocity
        self.control = np.zeros(3)

        self.mode = Mode.FORMATION

        self.path = [np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value]])]

    def update_state(self, control, dt):
        # Limit the control signals
        control_norm = np.linalg.norm(control)
        if control_norm > UMAX:
            control = control/control_norm*UMAX

        velocity = self.velocity + control*dt
        velocity_norm = np.linalg.norm(velocity)
        if velocity_norm > VMAX:
            velocity = velocity/velocity_norm*VMAX

        position = self.position + velocity*dt

        # Update state
        self.stamp += dt
        self.position = position
        self.velocity = velocity
        self.control = control
        self.path.append(np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value]]))

    def compute_control(self, robots, dt):
        v_mig = self.behavior_migration()
        v_obs = self.behavior_obstacle()
        v_col = self.behavior_collision(robots)
        v_form = self.behavior_formation(robots)
        v_rand = self.behavior_random()
        desired_velocity = v_mig + v_form + v_obs + v_col + v_rand

        desired_control = (desired_velocity - self.velocity)/dt
        self.update_state(desired_control, dt)

    def behavior_migration(self):
        return VREF*UREF
    
    def behavior_formation(self, robots):
        v_form = 0
        for i in range(NUM_ROBOT):
            v_form += (robots[i].position - self.position) - (TOPOLOGY[i,:] - TOPOLOGY[self.index,:])
        return W_form*v_form
    
    def behavior_obstacle(self):
        v_obs = 0
        for j in range(len(OBSTACLES)):
            obstacle = OBSTACLES[j]
            obs_point = nearest_point_to_obstacle(self.position[:2], obstacle)
            obs_rel = self.position - np.concatenate([obs_point,[self.position[2]]])
            obs_dis = np.linalg.norm(obs_rel)
            if obs_dis < ALERT_RADIUS:
                v_obs += 0.5*(1/obs_dis - 1/ALERT_RADIUS)/(obs_dis**2)*obs_rel/obs_dis
        return W_obs*v_obs
    
    def behavior_collision(self, robots):
        v_col = 0
        for i in range(NUM_ROBOT):
            if i == self.index:
                continue
            pos_rel = self.position - robots[i].position
            pos_dis = np.linalg.norm(pos_rel)
            if pos_dis < ALERT_RADIUS:
                v_col += 2*(1/pos_dis - 1/ALERT_RADIUS)/(pos_dis**2)*pos_rel/pos_dis
        return W_col*v_col

    def behavior_random(self):
        return W_rand*np.concatenate([np.random.rand(2),[0]])