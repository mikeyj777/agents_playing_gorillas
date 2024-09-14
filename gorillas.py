import math
import multiprocessing.pool
import random
from trajectory import trajectory_with_drag
import itertools
import numpy as np
import multiprocessing

from gorilla import Gorilla
from consts import Consts

class Gorillas_Game:
    '''
    
    gorilla game
    2-player game, but will train multiple gorillas
    reinforcement learning
    state:  distance, target_y, wind_speed.  one gorilla has the positive values associated with these.  
    the other is negative
    actions: angle (in degrees), and speed
    
    '''
    
    def __init__(self, num_players = 2):
        self.num_players = num_players
        self.players = []
        for _ in range(self.num_players):
            self.players.append(Gorilla())

    def get_y_at_closest_x(self, traj_dataset, distance):
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

    def start_the_action(self, player):
        player.get_action()


    def initialize_game(self):
        self.wind_speed = random.randint(0, 10)
        player_1 = random.randint(0, self.num_players-1)
        player_2 = random.randint(0, self.num_players-1)
        while player_2 == player_1:
            player_2 = random.randint(0, self.num_players-1)
        self.g1:Gorilla = self.players[player_1]
        self.g2:Gorilla = self.players[player_2]
        self.distance = abs(self.g1.x_coord - self.g2.x_coord)
        self.g1.target_y = self.g2.height - self.g1.height
        self.g1.state = {
            'distance': self.distance,
            'wind_speed': self.wind_speed,
            'target_y': self.g1.target_y,
        }
        self.g2.target_y = -self.g1.target_y
        self.g2.state = {
            'distance': -self.distance,
            'wind_speed': -self.wind_speed,
            'target_y': self.g2.target_y,
        }

    
    def play_game(self):
        keep_playing = True
        while keep_playing:
            self.initialize_game()
            self.players = [self.g1, self.g2]
            with multiprocessing.Pool() as p:
                _ = p.map(self.start_the_action, self.players)
