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
        self.scaling_factor = 1.0

        self.path = [np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value, self.scaling_factor]])]

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
        self.path.append(np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value, self.scaling_factor]]))

    def compute_control(self, robots, dt):
        v_mig = self.behavior_migration()
        v_obs = self.behavior_obstacle()
        v_col = self.behavior_collision(robots)

        self.mode_changing()
        if self.mode == Mode.FORMATION:
            v_form = self.behavior_formation(robots)
            desired_velocity = v_mig + v_form + v_obs + v_col
        else:
            v_tail = self.behavior_tailgating(robots)
            desired_velocity = v_mig + v_tail + v_obs + v_col

        desired_control = (desired_velocity - self.velocity)/dt
        self.update_state(desired_control, dt)

    def behavior_migration(self):
        return VREF*UREF
    
    def behavior_formation(self, robots):
        v_form = 0
        for i in range(NUM_ROBOT):
            v_form += (robots[i].position - self.position) - (TOPOLOGY[i,:] - TOPOLOGY[self.index,:])
        return W_form*v_form
    
    def behavior_tailgating(self, robots):
        leader_idx = self.select_leader(robots)
        if leader_idx == -1:
            return np.zeros(3)
        u_ref = robots[leader_idx].velocity/np.linalg.norm(robots[leader_idx].velocity)
        v_tail = (robots[leader_idx].position - self.position - DREF*u_ref) + robots[leader_idx].velocity
        return W_tail*v_tail
    
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

    def select_leader(self, robots):
        positions = []
        for i in range(NUM_ROBOT):
            positions.append(robots[i].position)
        positions = np.array(positions)
            
        vec = (positions[:,0]-self.position[0])*UREF[0] + (positions[:,1]-self.position[1])*UREF[1]
        vec[np.where(vec<=0)] = np.inf
        if np.all(np.isinf(vec)): # is leader
            return -1
        leader_index = np.argmin(vec)
        return leader_index
    
    def mode_changing(self):
        # Check obstacles
        if len(OBSTACLES) == 0:
            self.mode = Mode.FORMATION
            self.scaling_factor = 1.0
            return
        
        # Estimate width of environment
        we = self.estimate_environment_width()
        if we is None:
            self.mode = Mode.FORMATION
            self.scaling_factor = 1.0
            return

        # Mode selection
        if we <= ALPHA*ROBOT_RADIUS:
            self.mode = Mode.TAILGATING
            self.scaling_factor = -1
        else:
            # Estimate width of formation
            wf = self.estimate_formation_width()
            scaling_factor = 1.0
            if we - 2*ROBOT_RADIUS < wf:
                scaling_factor = (we - 2*ROBOT_RADIUS)/wf
            self.mode = Mode.FORMATION
            self.scaling_factor = scaling_factor

    def estimate_formation_width(self):
        y_left = np.min(TOPOLOGY[:,1])
        y_right = np.max(TOPOLOGY[:,1])
        return y_right - y_left
    
    def estimate_environment_width(self):
        obs_left = None; obs_right = None
        for j in range(len(OBSTACLES)):
            obstacle = OBSTACLES[j]
            obs_point = nearest_point_to_obstacle(self.position[:2], obstacle)
            # Check the nearest obstacle point is in the front of robot
            if np.dot((self.position[:2] - obs_point), UREF[:2]) <= 0:
                # Check the nearest obstacle point is in the left or right side of robot
                if np.dot((self.position[:2] - obs_point), np.array([UREF[1], UREF[0]])) < 0:
                    if obs_left is None or np.linalg.norm(self.position[:2] - obs_point) < np.linalg.norm(self.position[:2] - obs_left):
                        obs_left  = obs_point
                else:
                    if obs_right is None or np.linalg.norm(self.position[:2] - obs_point) < np.linalg.norm(self.position[:2] - obs_right):
                        obs_right = obs_point
        if obs_left is None or obs_right is None:
            return None
        theta = math.atan2(UREF[1], UREF[0])
        width = abs((obs_left[0]-obs_right[0])*np.sin(theta) + (obs_left[1]-obs_right[1])*np.cos(theta))
        return width
        