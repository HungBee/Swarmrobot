import numpy as np
import math
from config import *
from utils import polygon_area, polygon_centroid, nearest_point_to_obstacle


class ElasticMembrane:
    def __init__(self, center, radius):
        self.num_nodes = MEMBRANE_NODES
        self.center = center.copy()
        self.radius_ref = radius
        self.target_area = np.pi * radius * radius   # diện tích cố định
        self.a = radius       # bán trục lớn hiện tại
        self.b = radius       # bán trục nhỏ hiện tại
        self.theta = 0.0      # góc xoay (hướng di chuyển)
        angles = np.linspace(0, 2*np.pi, self.num_nodes, endpoint=False)
        self.nodes = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles)
        ])
        self.velocities = np.zeros_like(self.nodes)
        self.rest_length = 2 * np.pi * radius / self.num_nodes


    def estimate_environment_width(self, robot, obstacles):
        """Ước lượng chiều rộng khe hẹp phía trước robot (dùng logic giống robot)."""
        obs_left = None
        obs_right = None
        for obs in obstacles:
            near = nearest_point_to_obstacle(robot.position[:2], obs)
            rel = robot.position[:2] - near
            if np.dot(rel, UREF[:2]) <= 0:  # điểm nằm phía trước
                side = np.dot(rel, np.array([UREF[1], UREF[0]]))
                if side < 0:  # bên trái
                    if obs_left is None or np.linalg.norm(rel) < np.linalg.norm(robot.position[:2] - obs_left):
                        obs_left = near
                else:         # bên phải
                    if obs_right is None or np.linalg.norm(rel) < np.linalg.norm(robot.position[:2] - obs_right):
                        obs_right = near
        if obs_left is None or obs_right is None:
            return None
        theta = math.atan2(UREF[1], UREF[0])
        width = abs((obs_left[0]-obs_right[0])*np.sin(theta) + (obs_left[1]-obs_right[1])*np.cos(theta))
        return width


    def update(self, obstacles, robots, dt):
        n = self.num_nodes
        # Tâm bám trọng tâm đàn
        robot_centroid = np.mean([r.position[:2] for r in robots], axis=0)
        self.center += MEMBRANE_ALPHA_CENTER * (robot_centroid - self.center)


        # Chọn robot gần tâm nhất để cảm nhận khe
        closest_robot = min(robots, key=lambda r: np.linalg.norm(r.position[:2] - self.center))
        gap_width = self.estimate_environment_width(closest_robot, obstacles)


        # Bán kính tham chiếu hiện tại (nếu là circle) dựa trên vị trí robot
        max_dist = 0.0
        for r in robots:
            d = np.linalg.norm(r.position[:2] - self.center) + ROBOT_RADIUS
            if d > max_dist:
                max_dist = d
        target_radius = max_dist + 0.5
        self.radius_ref += MEMBRANE_RADIUS_EMA * (target_radius - self.radius_ref)
        self.radius_ref = max(self.radius_ref, ROBOT_RADIUS*2)


        # Xác định hình dạng mong muốn
        if MEMBRANE_GAP_RESPONSE and gap_width is not None and gap_width < 2 * self.radius_ref:
            # Chuyển sang ellipse
            b_target = max(ROBOT_RADIUS, (gap_width - 2*ROBOT_RADIUS)/2)
            a_target = self.target_area / (np.pi * b_target)
            self.a += 0.3 * (a_target - self.a)      # làm mượt
            self.b += 0.3 * (b_target - self.b)
            self.theta = math.atan2(UREF[1], UREF[0])
        else:
            # Trở về circle
            self.a += 0.3 * (self.radius_ref - self.a)
            self.b += 0.3 * (self.radius_ref - self.b)
            self.theta = 0.0


        # Tạo vị trí tham chiếu ellipse/circle
        ref_nodes = np.zeros((n, 2))
        for i in range(n):
            angle = 2 * np.pi * i / n
            dx = self.a * np.cos(angle)
            dy = self.b * np.sin(angle)
            rx = dx * math.cos(self.theta) - dy * math.sin(self.theta)
            ry = dx * math.sin(self.theta) + dy * math.cos(self.theta)
            ref_nodes[i] = self.center + np.array([rx, ry])


        # Lực đàn hồi và đẩy vật cản
        forces = np.zeros_like(self.nodes)
        total_obs_push = np.zeros(2)


        for i in range(n):
            vec_ref = ref_nodes[i] - self.nodes[i]
            forces[i] += MEMBRANE_REF_K * vec_ref
            j = (i+1) % n
            vec = self.nodes[j] - self.nodes[i]
            dist = np.linalg.norm(vec)
            if dist > 0:
                strain = (dist - self.rest_length) / self.rest_length
                force_mag = MEMBRANE_SPRING_K * strain / (1.0 + abs(strain))
                forces[i] += force_mag * (vec / dist)
                forces[j] -= force_mag * (vec / dist)
            for obs in obstacles:
                near = nearest_point_to_obstacle(self.nodes[i], obs)
                vec_obs = self.nodes[i] - near
                dist_obs = np.linalg.norm(vec_obs)
                if dist_obs < MEMBRANE_THRESHOLD and dist_obs > 1e-6:
                    penetration = MEMBRANE_THRESHOLD - dist_obs
                    force_mag = MEMBRANE_OBSTACLE_K * penetration
                    f = force_mag * (vec_obs / dist_obs)
                    forces[i] += f
                    total_obs_push += f


        # Đẩy tâm màng nếu có lực từ vật cản
        if np.linalg.norm(total_obs_push) > 0:
            self.center += MEMBRANE_CENTER_PUSH_K * total_obs_push / n * dt


        self.velocities = MEMBRANE_DAMPING * self.velocities + forces * dt
        vn = np.linalg.norm(self.velocities, axis=1)
        ex = vn > MEMBRANE_MAX_VEL
        if np.any(ex):
            self.velocities[ex] = (self.velocities[ex].T * (MEMBRANE_MAX_VEL / vn[ex])).T
        self.nodes += self.velocities * dt


        # Giới hạn độ lõm
        for i in range(n):
            vec = ref_nodes[i] - self.nodes[i]
            dist = np.linalg.norm(vec)
            if dist > MEMBRANE_MAX_INDENT:
                self.nodes[i] = ref_nodes[i] - (vec / dist) * MEMBRANE_MAX_INDENT


    def get_nodes(self):
        return self.nodes
