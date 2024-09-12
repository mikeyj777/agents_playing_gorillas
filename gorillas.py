import math
import random
from trajectory import trajectory_with_drag
import itertools
import numpy as np

from consts import Consts


def get_y_at_closest_x(traj_dataset, distance):
    min_dist = np.inf
    y_at_closest_x = -1
    closest_x = np.inf
    for _, row in traj_dataset.iterrows():
        x = row['position_x']
        y = row['position_y']
        curr_dist = abs(x-distance)
        if curr_dist < min_dist:
            min_dist = curr_dist
            y_at_closest_x = y
            closest_x = x
    
    if abs(closest_x - distance) > Consts.IMPACT_TOLERANCE:
        y_at_closest_x = None

    return y_at_closest_x

# Set random heights for both gorillas and distance between them
print('\r\n\n------------------\n\n')
g1_height = random.randint(1, 100)  # Gorilla 1 height (1 to 100 units)
g2_height = random.randint(1, 100)  # Gorilla 2 height (1 to 100 units)
max_distance = Consts.X_BOUNDS['max']
if math.isinf(max_distance):
    max_distance = 300
distance = random.randint(int(max_distance / 5), max_distance)  # Distance between gorillas (50 to 250 units)

# Set random wind speed and direction
wind_speed = random.randint(-10, 10)  # Wind (-10 to +10 units)

first_loop = True
# Main game loop
while True:
    # Display positions and wind conditions
    print(f"Gorilla 1 Height: {g1_height}")
    print(f"Gorilla 2 Height: {g2_height}")
    print(f"Distance between Gorillas: {distance}")
    print(f"Wind Speed: {wind_speed}")
    if not first_loop:
        print(f"Previous angle: {angle}")
        print(f"Previous speed: {speed}")
    first_loop = False

    # Player input
    angle = float(input("Input angle (degrees): "))
    speed = float(input("Input speed (units): "))
    speed = max(speed, Consts.MIN_SPEED)

    cos_angle = math.cos(math.radians(angle))
    sin_angle = math.sin(math.radians(angle))
    cos_veloc = speed * cos_angle
    sin_veloc = speed * sin_angle
    wind_speed = 0
    g1_height = 10

    print(f'Angle: {angle}, Speed: {speed}, cos_angle: {cos_angle}, sin_angle: {sin_angle}, cos_veloc: {cos_veloc}, sin_veloc: {sin_veloc}')

    # Convert angle to radians
    rad_angle = math.radians(angle)

    traj_dataset, shot_successful = trajectory_with_drag(start_speed=speed, angle_rad=rad_angle, start_height=g1_height, wind_speed=wind_speed, target_xy=(distance, g2_height))
    
    final_y = traj_dataset['position_y'].iloc[-1]
    final_x = traj_dataset['position_x'].iloc[-1]
    banana_height_near_enemy = get_y_at_closest_x(traj_dataset, distance)
    
    if shot_successful:
        print(f"Hit! The banana landed at height {final_y:.2f}, knocking the enemy gorilla down!")
        break
    else:
        if banana_height_near_enemy is None:
            print(f"Missed! The banana landed far from target at height {final_y:.2f}, distance {final_x:.2f}. Try again!")
            continue
        print(f"Missed! The banana passed the enemy gorilla at height {banana_height_near_enemy:.2f}. Try again!")
