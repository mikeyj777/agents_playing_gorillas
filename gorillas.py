import math
import random
from trajectory import trajectory_with_drag
import itertools
import numpy as np

from consts import Consts


def get_y_at_closest_x(traj_dataset, distance):
    min_dist = np.inf
    y_at_closest_x = -1
    for row in traj_dataset:
        x = row['position_x']
        y = row['position_y']
        curr_dist = abs(x-distance)
        if curr_dist < min_dist:
            min_dist = curr_dist
            y_at_closest_x = y
    
    return y_at_closest_x

# Set random heights for both gorillas and distance between them
print('\r\n\n------------------\n\n')
g1_height = random.randint(1, 100)  # Gorilla 1 height (1 to 100 units)
g2_height = random.randint(1, 100)  # Gorilla 2 height (1 to 100 units)
max_distance = Consts.X_BOUNDS['max']
distance = random.randint(int(max_distance / 5), max_distance)  # Distance between gorillas (50 to 250 units)

# Set random wind speed and direction
wind_speed = random.randint(-10, 10)  # Wind (-10 to +10 units)

# Display positions and wind conditions
print(f"Gorilla 1 Height: {g1_height}")
print(f"Gorilla 2 Height: {g2_height}")
print(f"Distance between Gorillas: {distance}")
print(f"Wind Speed: {wind_speed}")

# Gravity constant (for Earth, in meters per second squared)
gravity = 9.8

# Main game loop
while True:
    # Player input
    angle = float(input("Input angle (degrees): "))
    speed = float(input("Input speed (units): "))

    # Convert angle to radians
    rad_angle = math.radians(angle)

    traj_dataset = trajectory_with_drag(start_speed=speed, angle_rad=rad_angle, start_height=g1_height, wind_speed=wind_speed)
    
    banana_height = get_y_at_closest_x(traj_dataset, distance)

    # Check if the banana hits the other gorilla
    if abs(banana_height - g2_height) < 1.0:  # Allowing a small margin for hitting
        print(f"Hit! The banana landed at height {banana_height:.2f}, knocking the enemy gorilla down!")
        break
    else:
        print(f"Missed! The banana landed at height {banana_height:.2f}. Try again!")
