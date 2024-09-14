import uuid
import math
import random
from consts import Consts

from agent import Agent
from trajectory import trajectory_with_drag

class Gorilla:
    def __init__(self, upwind=True):
        self.id = f'{uuid.uuid4()}'
        self.height = random.randint(1, 100)  # Gorilla height (1 to 100 units)
        self.x_coord = random.randint(0, Consts.X_BOUNDS['max'])
        self.state = {}
        self.target_y = None
        self.agent = Agent(hyperparameter_set='gorillas', gorilla=self)
        self.upwind = upwind
    
    def get_action(self):

        self.agent.run(init_state=self.state, is_training=True)
    
    def step(self, action):
        angle = action[0]
        speed = action[1]
        wind_speed = self.state['wind_speed']
        distance = self.state['distance']


        g2_height = self.target_y

        rad_angle = math.radians(angle)
        
        traj_dataset, _ = trajectory_with_drag(start_speed=speed, angle_rad=rad_angle, start_height=0, wind_speed=wind_speed, target_xy=(distance, g2_height))
        
        final_y = traj_dataset['position_y'].iloc[-1]
        final_x = traj_dataset['position_x'].iloc[-1]

        return final_x, final_y

