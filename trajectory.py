import math
import numpy as np
import pandas as pd

from consts import Consts
from trajectory_plotter import plot_trajectory

def trajectory_with_drag(start_speed, angle_rad, start_height, wind_speed, time_step=None, target_xy=None):
    '''
    checks for impact with target

    '''
    g = Consts.GRAVITY
    cd = Consts.BANANA_DRAG_COEFFICIENT
    rho = Consts.AIR_DENSITY
    A = Consts.BANANA_CROSS_SECTIONAL_AREA
    m = Consts.BANANA_MASS
    v0 = start_speed
    
    if time_step is None:
        # use a time step that will result in horizontal spacing no greater than 99% of impact tolerance
        # time_step = 0.99 * Consts.IMPACT_TOLERANCE / start_speed
        time_step = 0.1
    
    if target_xy is None:
        target_xy = (np.inf, np.inf)

    x_target = target_xy[0]
    y_target = target_xy[1]

    # Initial conditions
    velocity_x = start_speed * math.cos(angle_rad)
    velocity_y = start_speed * math.sin(angle_rad)
    position_x = 0
    position_y = start_height
    u0 = velocity_x
    # Time simulation variables
    time = 0


    # Store the velocity at each step
    trajectory_data = [{
        'time': 0, 
        'position_x': position_x, 
        'position_y': position_y, 
        'velocity': math.sqrt(velocity_x**2 + velocity_y**2)
    }]

    shot_successful = False
    # Simulation loop
    time += time_step
    while position_y > 0 and not math.isinf(position_y):
        
        v = math.sqrt(velocity_x**2 + velocity_y**2)

        #terminal velocity
        vt = math.sqrt(2 * m * m / (cd * rho * A))

        
        position_y = vt**2 / (2*g) * math.log((v0 **2 + vt**2) / (v **2 + vt**2))
        position_x = vt**2 / (g) * math.log((vt**2 + g*u0*time) / (vt**2))

        # Calculate total velocity

        # if position_x > Consts.X_BOUNDS['max'] or position_x < Consts.X_BOUNDS['min'] or position_y > Consts.Y_BOUNDS['max']:
        #     position_y = np.inf
        #     break

        if position_y < 0:
            position_y = 0
        
        traj_data_row = {
            'time': time, 
            'position_x': position_x, 
            'position_y': position_y, 
            'velocity': v
        }

        # print(traj_data_row)
        
        # Store data
        trajectory_data.append(traj_data_row)
        
        # Update time
        time += time_step

        if abs(position_x - x_target) <= Consts.IMPACT_TOLERANCE and abs(position_y - y_target) <= Consts.IMPACT_TOLERANCE:
            shot_successful = True
            break

    trajectory_data_df = pd.DataFrame(trajectory_data)
    
    # plot_trajectory(trajectory_data_df)

    return trajectory_data_df, shot_successful



if __name__ == "__main__":

    # Example usage
    start_speed = 30  # m/s
    angle = 45  # degrees
    start_height = 10  # meters
    wind_speed = 2  # m/s

    final_speed, velocity_data = trajectory_with_drag(start_speed, angle, start_height, wind_speed)

    print(f"Final speed when hitting the ground: {final_speed:.2f} m/s")
