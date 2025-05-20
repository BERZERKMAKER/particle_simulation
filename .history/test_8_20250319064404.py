import pygame
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

# Initialize Pygame
pygame.init()

# Set the size of the window
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1200
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
PARTICLE_DETECTION_RANGE=18


# Set the PID parameters
k_p = 0.2
k_d = 0.3

k_p_2 = 0.02
k_d_2 = 0.01


# Set the particle swarm size
NUM_PARTICLES = 10

# Set the initial start and end points
start_point = np.array([250, 1000])  
end_point = np.array([1400, 200])  


# Create particles
def create_particles(start_point, num_particles, position_range=100, particle_radius=4):
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


particles = create_particles(start_point, NUM_PARTICLES)

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
wall_width = 1600
wall_height = 1200
wall_thickness = 10
edge_obstacles = [
   
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_width / 2, WINDOW_HEIGHT / 2 - wall_height / 2, wall_width, wall_thickness), 'color': BLUE},  # Top wall
   
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_width / 2, WINDOW_HEIGHT / 2 - wall_height / 2, wall_thickness, wall_height), 'color': BLUE},  # Left wall
      
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_width / 2, WINDOW_HEIGHT / 2 + wall_height / 2 - wall_thickness, wall_width, wall_thickness), 'color': BLUE},  # Bottom wall
    
    {'rect': pygame.Rect(WINDOW_WIDTH / 2 + wall_width / 2 - wall_thickness, WINDOW_HEIGHT / 2 - wall_height / 2, wall_thickness, wall_height), 'color': BLUE},  # Right wall
]

inner_obstacles= [

    {'rect': pygame.Rect(WINDOW_WIDTH / 2 - wall_thickness, WINDOW_HEIGHT / 2 - 200 , wall_thickness, 500), 'color': GREEN},  # Center wall
 
]

#creat force from obstacle
def point_to_line_segment_distance(point, wall_rect):
    """计算点到线段的最短距离"""
    if wall_rect.width < wall_rect.height:  # 竖直墙
        start = np.array([wall_rect.centerx, wall_rect.top])
        end = np.array([wall_rect.centerx, wall_rect.bottom])
    else:  # 水平墙
        start = np.array([wall_rect.left, wall_rect.centery])
        end = np.array([wall_rect.right, wall_rect.centery])
    
    wall_vector = end - start
    point_vector = point - start
    wall_length = np.linalg.norm(wall_vector)
    wall_unit = wall_vector / wall_length
    
    projection = np.dot(point_vector, wall_unit)
    
    if projection <= 0:
        closest = start
    elif projection >= wall_length:
        closest = end
    else:
        closest = start + wall_unit * projection
    
    return np.linalg.norm(point - closest), closest

def wall_repulsive_force(particle_pos, wall_rect, k_rep=8000.0, influence_dist=40.0):
    
    distance, closest_point = point_to_line_segment_distance(particle_pos, wall_rect)
    
    if distance < influence_dist:
        magnitude = k_rep * ( (influence_dist/distance) **2)
        direction = particle_pos - closest_point
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        return magnitude * direction
    return np.zeros(2)


# Create the window
window = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('Particle Simulation')

def calculate_average_velocity_xy(particles):
    velocities = np.array([particle['vel'] for particle in particles])
    return np.mean(velocities, axis=0)

# def calculate_average_position_xy(particles):
#     positions = np.array([particle['pos'] for particle in particles])
#     return np.mean(positions, axis=0)




def check_collision_with_obstacles(particle, edge_obstacles, restitution=0.8):
    particle_rect = pygame.Rect(particle['pos'][0] - PARTICLE_RADIUS,
                                particle['pos'][1] - PARTICLE_RADIUS,
                                2 * PARTICLE_RADIUS,
                                2 * PARTICLE_RADIUS)
    
    for obstacle in edge_obstacles:
        if obstacle['rect'].colliderect(particle_rect):
         
            if particle_rect.right >= obstacle['rect'].left and particle_rect.left < obstacle['rect'].left:
                particle['vel'][0] = -abs(particle['vel'][0]) * restitution  
                particle['pos'][0] = obstacle['rect'].left - PARTICLE_RADIUS  
            if particle_rect.left <= obstacle['rect'].right and particle_rect.right > obstacle['rect'].right:
                particle['vel'][0] = abs(particle['vel'][0]) * restitution   
                particle['pos'][0] = obstacle['rect'].right + PARTICLE_RADIUS  

            
            if particle_rect.bottom >= obstacle['rect'].top and particle_rect.top < obstacle['rect'].top:
                particle['vel'][1] = -abs(particle['vel'][1]) * restitution  
                particle['pos'][1] = obstacle['rect'].top - PARTICLE_RADIUS  
            if particle_rect.top <= obstacle['rect'].bottom and particle_rect.bottom > obstacle['rect'].bottom:
                particle['vel'][1] = abs(particle['vel'][1]) * restitution   
                particle['pos'][1] = obstacle['rect'].bottom + PARTICLE_RADIUS  
                



def custom_potential(r, desired_distance=15, k1=5.0, k2=2.0):
 
    # 使用平方项来确保势能在期望距离处最小
    return k1 * (r - desired_distance)**2 + k2 * (1/r)**4

def get_force(r, potential_func, desired_distance=15, k1=5.0, k2=2.0):
    """
    计算力（势能的负梯度）
    """
    dr = 0.0001  # 用于数值微分的小增量
    force = -(potential_func(r + dr, desired_distance, k1, k2) - 
             potential_func(r - dr, desired_distance, k1, k2)) / (2 * dr)
    return force

#cluster determine
def detect_particle_clusters_opencv(particles, detection_range=18, window_height=1200, window_width=1600):

    positions = np.array([particle['pos'] for particle in particles])

    for pos in positions:
        if pos[0] < 0 or pos[0] >= window_width or pos[1] < 0 or pos[1] >= window_height:
            raise ValueError(f"Particle position {pos} is out of bounds for the given window size.")

    image = np.zeros((window_height, window_width), dtype=np.uint8)

    
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(image, (x, y), radius=18, color=100, thickness=-1)

    
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
    image = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH), dtype=np.uint8)
    positions = np.array([particle['pos'] for particle in particles])
    for pos in positions:
        x, y = int(pos[0]), int(pos[1])
        cv2.circle(image, (x, y), radius=PARTICLE_DETECTION_RANGE, color=100, thickness=-1)
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
                                        k1=10, k2=30)
                
                # 力的方向是两个粒子的连线方向
                force = force_magnitude * direction
                total_force += force
            else:
                force = 0
                total_force += force
        
        # 2. 添加中央墙的斥力
        for obstacle in inner_obstacles:
            wall_force = wall_repulsive_force(particle['pos'], obstacle['rect'])
            total_force += wall_force

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

        # 限制速度到最大速度
        speed = np.linalg.norm(particle['vel'])
        if speed > PARTICLE_MAX_SPEED:
            particle['vel'] = (particle['vel'] / speed) * PARTICLE_MAX_SPEED
            # total_force=0.001*total_force/np.linalg.norm(total_force)
        
        # 输出更新后的速度信息
        # print(f"Particle {i} - Updated Velocity: {particle['vel']}")

        # 根据速度更新位置
        particle['pos'] += particle['vel']
        
 
    window.fill(BLACK)
    


    for particle in particles:
        pygame.draw.circle(window, PARTICLE_COLOR, (int(particle['pos'][0]), int(particle['pos'][1])), PARTICLE_RADIUS)

    for obstacle in edge_obstacles:
        pygame.draw.rect(window, obstacle['color'], obstacle['rect'])
        
    for obstacle in inner_obstacles:
        pygame.draw.rect(window, obstacle['color'], obstacle['rect'])

    pygame.draw.circle(window, WHITE, (int(start_point[0]), int(start_point[1])), 10)
    pygame.draw.circle(window, WHITE, (int(end_point[0]), int(end_point[1])), 10)

    pygame.display.flip()

    clock.tick(60)

pygame.quit()