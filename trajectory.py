import math
import numpy as np

from consts import Consts
from plotter import plot_trajectory

def trajectory_with_drag(start_speed, angle_rad, start_height, wind_speed, time_step=None, target_xy=None):
    '''
    checks for impact with target

    '''
    if time_step is None:
        time_step = Consts.TRAJECTORY_TIME_STEP
    
    if target_xy is None:
        target_xy = (np.inf, np.inf)

    x_target = target_xy[0]
    y_target = target_xy[1]

    # Initial conditions
    velocity_x = start_speed * math.cos(angle_rad)
    velocity_y = start_speed * math.sin(angle_rad)
    position_x = 0
    position_y = start_height

    # Time simulation variables
    time = 0

    # Time simulation variables
    time = 0

    # Store the velocity at each step
    trajectory_data = []

    # Simulation loop
    while position_y > 0 and not math.isinf(position_y):
        # Calculate total velocity
        velocity_total = math.sqrt(velocity_x**2 + velocity_y**2)
        
        # Drag force
        drag_force = 0.5 * Consts.BANANA_DRAG_COEFFICIENT * Consts.AIR_DENSITY * Consts.BANANA_CROSS_SECTIONAL_AREA * velocity_total**2
        
        # Drag acceleration
        drag_accel_x = (drag_force / Consts.BANANA_MASS) * (velocity_x / velocity_total)
        drag_accel_y = (drag_force / Consts.BANANA_MASS) * (velocity_y / velocity_total)
        
        # Update velocities
        velocity_x -= drag_accel_x * time_step
        velocity_y -= (drag_accel_y + Consts.GRAVITY) * time_step
        
        # Wind effect (wind only affects x component)
        velocity_x += wind_speed * time_step
        
        # Update positions
        position_y += velocity_y * time_step
        position_x += velocity_x * time_step

        if position_x > Consts.X_BOUNDS['max'] or position_x < Consts.X_BOUNDS['min']:
            position_y = np.inf

        if position_y < 0:
            position_y = 0
            velocity_x = 0
            velocity_y = 0
        
        traj_data_row = {
            'time': time, 
            'position_x': position_x, 
            'position_y': position_y, 
            'velocity_x': velocity_x, 
            'velocity_y': velocity_y,
            'velocity': math.sqrt(velocity_x**2 + velocity_y**2)
        }

        print(traj_data_row)
        
        # Store data
        trajectory_data.append(traj_data_row)
        
        # Update time
        time += time_step

        if abs(position_x - x_target) <= Consts.IMPACT_TOLERANCE and abs(position_y - y_target) <= Consts.IMPACT_TOLERANCE:
            break

    plot_trajectory(trajectory_data)

    return trajectory_data



if __name__ == "__main__":

    # Example usage
    start_speed = 30  # m/s
    angle = 45  # degrees
    start_height = 10  # meters
    wind_speed = 2  # m/s

    final_speed, velocity_data = trajectory_with_drag(start_speed, angle, start_height, wind_speed)

    print(f"Final speed when hitting the ground: {final_speed:.2f} m/s")
