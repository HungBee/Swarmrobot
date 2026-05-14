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
        self.path.append(np.concatenate([[self.stamp], self.position, self.velocity, self.control, [self.mode.value, self.scaling_factor]]))

    def compute_control(self, robots, dt, membrane=None):
        v_mig = self.behavior_migration()
        v_obs = self.behavior_obstacle()
        v_col = self.behavior_collision(robots)

        self.mode_changing(robots)
        if self.mode == Mode.FORMATION:
            v_form = self.behavior_formation(robots)
            v_collective = self.behavior_collective_avoid(robots)  # chỉ tập thể khi formation
            desired_velocity = v_mig + v_form + v_obs + v_col + v_collective
        else:
            v_tail = self.behavior_tailgating(robots)
            desired_velocity = v_mig + v_tail + v_obs + v_col   # không tập thể khi tailgating

        # Lực màng (rất nhẹ)
        if membrane is not None:
            v_memb = self.behavior_membrane(membrane)
            desired_velocity += v_memb

        desired_control = (desired_velocity - self.velocity) / dt
        self.update_state(desired_control, dt)

    def behavior_migration(self):
        return VREF * UREF

    def behavior_formation(self, robots):
        v_form = np.zeros(3)
        for i in range(NUM_ROBOT):
            v_form += (robots[i].position - self.position) - (TOPOLOGY[i,:] - TOPOLOGY[self.index,:])
        return W_form * v_form

    def behavior_tailgating(self, robots):
        leader_idx = self.select_leader(robots)
        if leader_idx == -1:
            return np.zeros(3)
        u_ref = robots[leader_idx].velocity / (np.linalg.norm(robots[leader_idx].velocity) + 1e-6)
        v_tail = (robots[leader_idx].position - self.position - DREF * u_ref) + robots[leader_idx].velocity
        return v_tail

    def behavior_obstacle(self):
        v_obs = np.zeros(3)
        for obs in OBSTACLES:
            near = nearest_point_to_obstacle(self.position[:2], obs)
            rel = self.position - np.concatenate([near, [self.position[2]]])
            d = np.linalg.norm(rel)
            if d < ALERT_RADIUS:
                v_obs += 0.5 * (1/d - 1/ALERT_RADIUS) / (d**2) * rel / d
        return W_obs * v_obs

    def behavior_collision(self, robots):
        v_col = np.zeros(3)
        for i in range(NUM_ROBOT):
            if i == self.index:
                continue
            rel = self.position - robots[i].position
            d = np.linalg.norm(rel)
            if d < ALERT_RADIUS:
                v_col += 2*(1/d - 1/ALERT_RADIUS) / (d**2) * rel / d
        return W_col * v_col

    def behavior_collective_avoid(self, robots):
        total_push = np.zeros(2)
        count = 0
        for r in robots:
            push = np.zeros(2)
            for obs in OBSTACLES:
                near = nearest_point_to_obstacle(r.position[:2], obs)
                d = np.linalg.norm(r.position[:2] - near)
                if d < ALERT_RADIUS:
                    push += (r.position[:2] - near) / d
            if np.linalg.norm(push) > 0:
                total_push += push / np.linalg.norm(push)
                count += 1
        if count > 0:
            avg_dir = total_push / count
            return np.concatenate([W_collective_avoid * avg_dir, [0]])
        return np.zeros(3)

    def select_leader(self, robots):
        positions = np.array([r.position for r in robots])
        vec = (positions[:,0] - self.position[0]) * UREF[0] + (positions[:,1] - self.position[1]) * UREF[1]
        vec[np.where(vec <= 0)] = np.inf
        if np.all(np.isinf(vec)):
            return -1
        return np.argmin(vec)

    def mode_changing(self, robots):
        if len(OBSTACLES) == 0:
            self.mode = Mode.FORMATION
            self.scaling_factor = 1.0
            return
        we = self.estimate_environment_width(robots)
        if we is None:
            self.mode = Mode.FORMATION
            self.scaling_factor = 1.0
            return
        if we <= ALPHA * ROBOT_RADIUS:
            self.mode = Mode.TAILGATING
            self.scaling_factor = -1
        else:
            wf = self.estimate_formation_width(robots)
            scaling_factor = 1.0
            if we - 2*ROBOT_RADIUS < wf:
                scaling_factor = (we - 2*ROBOT_RADIUS) / wf
            self.mode = Mode.FORMATION
            self.scaling_factor = scaling_factor

    def estimate_formation_width(self, robots):
        y_left = np.min(TOPOLOGY[:,1])
        y_right = np.max(TOPOLOGY[:,1])
        return y_right - y_left

    def estimate_environment_width(self, robots):
        obs_left = None; obs_right = None
        for obs in OBSTACLES:
            obstacle = obs
            obs_point = nearest_point_to_obstacle(self.position[:2], obstacle)
            if np.dot((self.position[:2] - obs_point), UREF[:2]) <= 0:
                if np.dot((self.position[:2] - obs_point), np.array([UREF[1], UREF[0]])) < 0:
                    if obs_left is None or np.linalg.norm(self.position[:2] - obs_point) < np.linalg.norm(self.position[:2] - obs_left):
                        obs_left = obs_point
                else:
                    if obs_right is None or np.linalg.norm(self.position[:2] - obs_point) < np.linalg.norm(self.position[:2] - obs_right):
                        obs_right = obs_point
        if obs_left is None or obs_right is None:
            return None
        theta = math.atan2(UREF[1], UREF[0])
        width = abs((obs_left[0]-obs_right[0])*np.sin(theta) + (obs_left[1]-obs_right[1])*np.cos(theta))
        return width

    def behavior_membrane(self, membrane):
        v_memb = np.zeros(3)
        nodes = membrane.get_nodes()
        if not point_inside_polygon(self.position[:2], nodes):
            near = nearest_point_on_polygon(self.position[:2], nodes)
            vec = np.array([near[0] - self.position[0], near[1] - self.position[1], 0.0])
            dist = np.linalg.norm(vec)
            if dist > 0:
                v_memb[:2] = W_membrane * (vec[:2] / dist) * min(dist, 1.5)
        return v_memb
