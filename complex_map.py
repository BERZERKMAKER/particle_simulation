import pygame
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

# Initialize Pygame
pygame.init()

# Set the size of the window
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Set colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Set the color of particles
PARTICLE_COLOR = RED
PARTICLE_RADIUS = 4
PARTICLE_MAX_SPEED = 4
PARTICLE_DESIRED_DISTANCE=15
PARTICLE_DETECTION_RANGE=20


# Set the PID parameters
k_p = 0.010
k_d = 0.015

k_p_2 = 0.4
k_d_2 = 0.01


# Set the particle swarm size
NUM_PARTICLES = 2

# Set the initial start and end points
start_point = np.array([250, 800])  
end_point = np.array([1400, 200])  


# Create particles
def create_particles(start_point, num_particles, position_range=80, particle_radius=4):
    particles = []
    
    for _ in range(num_particles):
        while True:
            
            new_pos = np.random.uniform(start_point - position_range, start_point + position_range, size=(2,))
            
           
            is_overlapping = False
            for particle in particles:
                existing_pos = particle['pos']
                distance = np.linalg.norm(new_pos - existing_pos)
                
                if distance < 2 * particle_radius:  
                    is_overlapping = True
                    break
            
            
            if not is_overlapping:
                particles.append({'pos': new_pos, 'vel': np.zeros(2)})
                break
    
    return particles





def get_frame_from_pygame(window):
    """Convert Pygame surface to OpenCV image"""
    view = pygame.surfarray.array3d(window)
    view = view.transpose([1, 0, 2])
    return cv2.cvtColor(view, cv2.COLOR_RGB2BGR)

def detect_particles(frame):
    """Detect red particles in the frame"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for red color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([160, 100, 100])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask1 + mask2
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get centroids of particles
    particle_positions = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            particle_positions.append(np.array([cx, cy]))
    
    return particle_positions

#creat obstacles
wall_width = WINDOW_WIDTH
wall_height = WINDOW_HEIGHT
wall_thickness = 10
edge_obstacles = [
   
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_width / 2, WINDOW_HEIGHT / 2 - wall_height / 2, wall_width, wall_thickness), 'color': BLUE},  # Top wall
   
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_width / 2, WINDOW_HEIGHT / 2 - wall_height / 2, wall_thickness, wall_height), 'color': BLUE},  # Left wall
      
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_width / 2, WINDOW_HEIGHT / 2 + wall_height / 2 - wall_thickness, wall_width, wall_thickness), 'color': BLUE},  # Bottom wall
    
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 + wall_width / 2 - wall_thickness, WINDOW_HEIGHT / 2 - wall_height / 2, wall_thickness, wall_height), 'color': BLUE},  # Right wall
]

def scale_polygon(points, scale=math.sqrt(3)):
    """按比例放大多边形，保持形心不变"""
    centroid = np.mean(points, axis=0)
    scaled = []
    for p in points:
        direction = p - centroid
        scaled_point = centroid + direction * scale
        scaled.append(tuple(scaled_point))
    return scaled
triangle_points = [(1000, 500), (1300, 850), (1100, 500)]
hexagon_points = [(200, 800), (450, 780), (500, 800),
                  (500, 840), (450, 860), (200, 840)]
inner_obstacles = [
    # 原始中心墙（矩形）
    {'type': 'rect', 'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_thickness,
                                         WINDOW_HEIGHT / 2 - 400,
                                         wall_thickness, 500), 'color': BLUE},

    # 新增圆形障碍
    {'type': 'circle', 'center': (600, 600), 'radius': 150, 'color': BLUE},

    # 新增三角形障碍
     {'type': 'polygon', 'points': scale_polygon(np.array(triangle_points)), 'color': BLUE},

    {'type': 'polygon', 'points': scale_polygon(np.array(hexagon_points)), 'color': BLUE},
]

#find the contours of the obstacles
def get_all_obstacle_contours(inner_obstacles, width, height):
    obstacle_mask = np.zeros((height, width), dtype=np.uint8)

    for obs in inner_obstacles:
        if obs['type'] == 'rect':
            pygame_rect = obs['rect']
            x, y, w, h = pygame_rect.left, pygame_rect.top, pygame_rect.width, pygame_rect.height
            cv2.rectangle(obstacle_mask, (x, y), (x + w, y + h), 255, thickness=-1)

        elif obs['type'] == 'circle':
            center = tuple(map(int, obs['center']))
            radius = int(obs['radius'])
            cv2.circle(obstacle_mask, center, radius, 255, thickness=-1)

        elif obs['type'] == 'polygon':
            points = np.array(obs['points'], dtype=np.int32)
            cv2.fillPoly(obstacle_mask, [points], 255)

    contours, _ = cv2.findContours(obstacle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

#creat force from obstacle
def unified_wall_repulsion(particle_pos, contours, k_rep=800000.0, influence_dist=100.0):
    total_force = np.zeros(2)
  

    for contour in contours:
        # 计算粒子到当前障碍物轮廓的最小距离
        contour_points = contour.reshape(-1, 2)
        
        distance = cv2.pointPolygonTest(contour_points, tuple(particle_pos), True)

        if abs(distance) < influence_dist and abs(distance) > 1e-2:
            # 计算法向方向（用质心近似）
            centroid = np.mean(contour.reshape(-1, 2), axis=0)
            direction = particle_pos - centroid
            direction = direction / (np.linalg.norm(direction) + 1e-5)

            # 计算反推回来的最近点
            closest_point = particle_pos - direction * distance

            # 计算斥力
            magnitude = k_rep * ((influence_dist / abs(distance)) ** 2)
            repel = magnitude * (particle_pos - closest_point)
            repel = repel / (np.linalg.norm(repel) + 1e-5)

            total_force += repel
            
            print(f"Repel force detected! Distance: {distance:.2f}, Magnitude: {magnitude:.2f}, Force: {repel}")

    return total_force

particles = create_particles(start_point, NUM_PARTICLES)
contours = get_all_obstacle_contours(inner_obstacles, WINDOW_WIDTH, WINDOW_HEIGHT)
# Create the window
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Particle Simulation')

def calculate_average_velocity_xy(particles):
    velocities = np.array([particle['vel'] for particle in particles])
    return np.mean(velocities, axis=0)




def check_collision_with_obstacles(particle, edge_obstacles, restitution=0.0):
    x, y = particle['pos']
    vx, vy = particle['vel']
    
    # 限制在屏幕范围内，防止越界
    if x - 2* PARTICLE_RADIUS < 0:
        particle['pos'][0] = PARTICLE_RADIUS
        particle['vel'][0] = abs(vx) * restitution
    elif x + 2 * PARTICLE_RADIUS > WINDOW_WIDTH:
        particle['pos'][0] = WINDOW_WIDTH - PARTICLE_RADIUS
        particle['vel'][0] = -abs(vx) * restitution

    if y - 2* PARTICLE_RADIUS < 0:
        particle['pos'][1] = 2* PARTICLE_RADIUS
        particle['vel'][1] = abs(vy) * restitution
    elif y + 2* PARTICLE_RADIUS > WINDOW_HEIGHT:
        particle['pos'][1] = WINDOW_HEIGHT - 2* PARTICLE_RADIUS
        particle['vel'][1] = -abs(vy) * restitution

    # 增强和内墙的碰撞检测（可选）
    particle_rect = pygame.Rect(x - PARTICLE_RADIUS, y - PARTICLE_RADIUS, 2 * PARTICLE_RADIUS, 2 * PARTICLE_RADIUS)
    for obstacle in edge_obstacles:
        if obstacle['rect'].colliderect(particle_rect):
            # 简单反弹处理
            if abs(particle['pos'][0] - obstacle['rect'].left) < PARTICLE_RADIUS or abs(particle['pos'][0] - obstacle['rect'].right) < PARTICLE_RADIUS:
                particle['vel'][0] = -vx * restitution
            if abs(particle['pos'][1] - obstacle['rect'].top) < PARTICLE_RADIUS or abs(particle['pos'][1] - obstacle['rect'].bottom) < PARTICLE_RADIUS:
                particle['vel'][1] = -vy * restitution



def custom_potential(r, desired_distance=15, k1=14.0, k2=2.0):
 
    # 使用平方项来确保势能在期望距离处最小
    return k1 * (r - desired_distance)**2 + k2 * (1/r)**2

def get_force(r, potential_func, desired_distance=15, k1=5.0, k2=3.0):
    """
    计算力（势能的负梯度）
    """
    dr = 0.0001  # 用于数值微分的小增量
    force = -(potential_func(r + dr, desired_distance, k1, k2) - 
             potential_func(r - dr, desired_distance, k1, k2)) / (2 * dr)
    return force

#cluster determine
def detect_particle_clusters_opencv(particles, detection_range=9, window_height=1200, window_width=1600):

    positions = np.array([particle['pos'] for particle in particles])

    for pos in positions:
        if pos[0] < 0 or pos[0] >= window_width or pos[1] < 0 or pos[1] >= window_height:
            raise ValueError(f"Particle position {pos} is out of bounds for the given window size.")

    image = np.zeros((window_height, window_width), dtype=np.uint8)

    
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(image, (x, y), radius=20, color=100, thickness=-1)

    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    
    num_clusters = num_labels - 1

    
    particle_labels = []
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        particle_labels.append(labels[y, x])
    
    cluster_centroids = centroids[1:]

    return num_clusters, particle_labels , cluster_centroids
CLUSTER_COLORS = {}

def get_cluster_color(label, existing_colors):
  
    
    if label not in existing_colors:
        # 为新的标签生成固定的颜色
        existing_colors[label] = (
            np.random.randint(100, 255),
            np.random.randint(100, 255),
            np.random.randint(100, 255)
        )
    return existing_colors[label]


# Main loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 使用OpenCV检测聚类
    num_clusters, labels, centroids = detect_particle_clusters_opencv(particles)
    print(f"Number of Clusters Detected: {num_clusters}")
    
    if num_clusters == 1:
        # 单个聚类，检查 centroids 数组
        if len(centroids) > 1:
            cluster_centroid = centroids[1]  # 索引 1 是聚类的质心
        else:
            # 如果只有一个聚类且只有背景质心，则赋予默认值
            cluster_centroid = centroids[0]  # 此时可能是背景质心，给默认值
    elif num_clusters > 1:
        # 多个聚类，计算所有聚类质心的平均值
        cluster_centroid = np.mean(centroids[1:], axis=0)  # 忽略背景质心
    else:
        # 没有聚类时，设置默认质心
        cluster_centroid = np.array([0, 0])

    
        # 显示二值图像
    debug_img = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
    cv2.drawContours(debug_img, contours, -1, 255, 1)
    cv2.imshow("Obstacle Contours", debug_img)
    image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
    positions = np.array([particle['pos'] for particle in particles])
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(image, (x, y), radius=20, color=100, thickness=-1)
    cv2.imshow('Binary Image', image)

    # 显示彩色连通区域图像
    colored_labels = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
    for label in range(1, num_clusters + 1):
        mask = (labels == label)  # 转置以匹配图像坐标系
        colored_labels[mask] = np.random.randint(0, 255, 3)
    cv2.imshow('Connected Components', colored_labels)
    cv2.waitKey(1)  # 用于更新图像显示
    # 为新的聚类标签分配颜色
    for label in set(labels):
        if label != 0:  # 排除背景标签
            get_cluster_color(label, CLUSTER_COLORS)

    # 可视化聚类（可选）
    if num_clusters > 0:
        color_labels = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
        for label in range(1, num_clusters + 1):
            color = np.random.randint(0, 255, 3).tolist()
            # 使用 numpy 的布尔索引
            mask = (labels == label)
            color_labels[mask] = color
        
        cv2.imshow('Particle Clusters', color_labels)
        cv2.waitKey(1)
        
    average_velocity = calculate_average_velocity_xy(particles)

 


    desired_acceleration_1 = k_p * (end_point - cluster_centroid) - k_d * average_velocity
 
    print(f"PD Acceleration (desired_acceleration_1): {desired_acceleration_1}")


    if cluster_centroid[0] > WINDOW_WIDTH / 2:
        if cluster_centroid[1] < WINDOW_HEIGHT / 2:
            desired_acceleration_2 = k_p_2 * (np.array([WINDOW_WIDTH - wall_thickness, wall_thickness])- cluster_centroid) - k_d_2 * average_velocity 
        else:
            desired_acceleration_2 = k_p_2 * (np.array([WINDOW_WIDTH - wall_thickness, WINDOW_HEIGHT-wall_thickness])- cluster_centroid) - k_d_2 * average_velocity 
    else:
        if cluster_centroid[1] < WINDOW_HEIGHT / 2:
            desired_acceleration_2 = k_p_2 * (np.array([wall_thickness,wall_thickness])-cluster_centroid) - k_d_2 * average_velocity 
        else:
            desired_acceleration_2 = k_p_2 * (np.array([wall_thickness,WINDOW_HEIGHT-wall_thickness])-cluster_centroid) - k_d_2 * average_velocity 
            
            
    print(f"PD Acceleration (desired_acceleration_2): {desired_acceleration_2}")

        # Step 1: 优先处理与障碍物的碰撞
    for particle in particles:
            # 检查与障碍物的碰撞，并立即调整速度和位置
            check_collision_with_obstacles(particle, edge_obstacles)
            
    forces = {i: np.zeros(2) for i in range(len(particles))}

        # Step 3: 计算所有粒子的合力
    for i, particle in enumerate(particles):
        # 初始化合力为零
        total_force = np.zeros(2)
        
        for j, other_particle in enumerate(particles):
            if i == j:
                continue  # 跳过自身

            # 计算两粒子之间的距离和方向
            distance = np.linalg.norm(particle['pos'] - other_particle['pos'])
            if distance == 0:
                continue  # 防止除以零
            
            direction = (particle['pos'] - other_particle['pos']) / distance
            
            # 根据距离判断力的类型并进行计算
            if distance <= 18:
                # 计算力的大小
                force_magnitude = get_force(distance, custom_potential, 
                                        desired_distance=14,
                                        k1=2, k2=0.2)
                
                # 力的方向是两个粒子的连线方向
                force = force_magnitude * direction
                total_force += force
            else:
                force = 0
                total_force += force
        
        # wall_force = unified_wall_repulsion(particle['pos'], contours)
        # total_force += wall_force

        # 3. 添加全局控制力
        if num_clusters > 1:
            total_force += desired_acceleration_2 
        elif num_clusters == 1:
            total_force += desired_acceleration_1 


        
        # 将合力存储在 forces 字典中
        forces[i] = total_force
        
        # Step 4: 更新所有粒子的速度和位置
    for i, particle in enumerate(particles):
        # 使用合力计算加速度（假设质量为 30）
        acceleration = forces[i]/30

        particle['vel'] += acceleration
        
        particle['vel'] *= 0.97  # 每帧速度衰减 2%


        # # 限制速度到最大速度
        # speed = np.linalg.norm(particle['vel'])
        # if speed > PARTICLE_MAX_SPEED:
        #     particle['vel'] = (particle['vel'] / speed) * PARTICLE_MAX_SPEED
            # total_force=0.001*total_force/np.linalg.norm(total_force)
            
        # 检查聚类数量和质心位置
        if num_clusters == 1:
            # 设置一个误差范围，允许质心接近目标点而不需要完全相等
            distance_to_target = np.linalg.norm(cluster_centroid - end_point)
            if distance_to_target < 5:  # 误差范围可调，例如 5 像素
                print(f"Cluster centroid reached the target point. Stopping particles.")
                # 将所有粒子的速度设为 0
                for particle in particles:
                    particle['vel'] = np.zeros(2)

        # 如果未达到目标点，则正常更新粒子运动
        else:
            # 计算全局控制力
            if num_clusters > 1:
                total_force += desired_acceleration_2
            elif num_clusters == 1:
                total_force += desired_acceleration_1
        
        # 输出更新后的速度信息
        # print(f"Particle {i} - Updated Velocity: {particle['vel']}")

        # 根据速度更新位置
        particle['pos'] += particle['vel']
        
 
    window.fill(BLACK)
    


    for particle in particles:
        pygame.draw.circle(window, PARTICLE_COLOR, (int(particle['pos'][0]), int(particle['pos'][1])), PARTICLE_RADIUS)

    for obstacle in edge_obstacles:
        pygame.draw.rect(window, obstacle['color'], obstacle['rect'])
        
    # for obstacle in inner_obstacles:
    #     if obstacle['type'] == 'rect':
    #         pygame.draw.rect(window, obstacle['color'], obstacle['rect'])
    #     elif obstacle['type'] == 'circle':
    #         pygame.draw.circle(window, obstacle['color'], obstacle['center'], obstacle['radius'])
    #     elif obstacle['type'] == 'polygon':
    #         pygame.draw.polygon(window, obstacle['color'], obstacle['points'])

    pygame.draw.circle(window, WHITE, (int(start_point[0]), int(start_point[1])), 10)
    pygame.draw.circle(window, WHITE, (int(end_point[0]), int(end_point[1])), 10)

    pygame.display.flip()

    clock.tick(30)

pygame.quit()