import numpy as np
from config import *
from utils import *
if CONTROLLER == 'erc':
    from robot_erc import Robot


def compute_virtual_region(robots, A_ref, prev_region=None, alpha=0.12):
    # Tâm thực tế hiện tại
    positions = np.array([r.position[:2] for r in robots])
    center_raw = np.mean(positions, axis=0)

    # Lấy chiều rộng khe hẹp (từ robot 0, như trong robot_erc)
    we = robots[0].estimate_environment_width()

    r_ref = np.sqrt(A_ref / np.pi)

    # Mục tiêu mode và kích thước
    if we is not None and we < 2 * r_ref:
        target_mode = 'ellipse'
        target_b = max(we / 2.0, ROBOT_RADIUS)
        target_a = A_ref / (np.pi * target_b)
        target_theta = np.arctan2(UREF[1], UREF[0])
    else:
        target_mode = 'circle'
        target_a = r_ref
        target_b = r_ref
        target_theta = 0.0

    # Nếu chưa có trạng thái trước (lần đầu), khởi tạo trực tiếp
    if prev_region is None:
        return {
            'center': center_raw,
            'mode': target_mode,
            'a': target_a,
            'b': target_b,
            'theta': target_theta
        }

    # Làm mịn EMA
    new_center = prev_region['center'] + alpha * (center_raw - prev_region['center'])
    new_a = prev_region['a'] + alpha * (target_a - prev_region['a'])
    new_b = prev_region['b'] + alpha * (target_b - prev_region['b'])
    delta_theta = np.arctan2(np.sin(target_theta - prev_region['theta']),
                             np.cos(target_theta - prev_region['theta']))
    new_theta = prev_region['theta'] + alpha * delta_theta

    return {
        'center': new_center,
        'mode': target_mode,
        'a': new_a,
        'b': new_b,
        'theta': new_theta
    }


def collision(robots):
    for i in range(NUM_ROBOT):
        position = robots[i].position
        for j in range(len(OBSTACLES)):
            obstacle = OBSTACLES[j]
            obs_point = nearest_point_to_obstacle(position[:2], obstacle)
            if np.linalg.norm(position[:2]-obs_point) < ROBOT_RADIUS:
                print("Obstacle")
                return True
    for i in range(NUM_ROBOT-1):
        pi = robots[i].position
        for j in range(i+1, NUM_ROBOT):
            pj = robots[j].position
            if np.linalg.norm(pi - pj) < 2*ROBOT_RADIUS:
                print("Inter-agent")
                return True
    return False


if __name__ == "__main__":
    # Tạo robot
    robots = []
    for i in range(NUM_ROBOT):
        robot = Robot(i, INITS[i,:])
        robots.append(robot)

    # Diện tích cố định từ bán kính vừa khít ban đầu
    A_ref = np.pi * INIT_RADIUS ** 2
    print(f"Diện tích cố định: {A_ref:.2f} m², bán kính tham chiếu: {INIT_RADIUS:.2f} m")

    # --- KHỞI TẠO VÙNG ẢO BAN ĐẦU (frame 0) ---
    # Dùng vị trí khởi tạo để region đầu tiên tiếp xúc với robot xa nhất
    region_initial = {
        'center': INIT_CENTER.copy(),
        'mode': 'circle',
        'a': INIT_RADIUS,
        'b': INIT_RADIUS,
        'theta': 0.0
    }
    region_history = [region_initial]
    prev_region = region_initial

    iter = 0
    while iter < ITER_MAX:
        if iter % 50 == 0:
            print(f"Iteration {iter}")

        # Điều khiển robot (code gốc của bạn)
        for i in range(NUM_ROBOT):
            robots[i].compute_control(robots, dt=TIMESTEP)

        # Cập nhật vùng ảo (EMA)
        region = compute_virtual_region(robots, A_ref, prev_region, alpha=0.12)
        region_history.append(region)
        prev_region = region

        # Kiểm tra va chạm
        if collision(robots):
            print("Va chạm! Dừng.")
            break

        # Kiểm tra đến đích
        if all(r.position[0] > XGOAL for r in robots):
            print("Đã đến đích!")
            break

        iter += 1

    # Lưu dữ liệu
    import pickle
    with open(FILE_NAME, 'wb') as file:
        data = [{'path': np.array(r.path)} for r in robots]
        data.append({'region_history': region_history})
        pickle.dump(data, file)
